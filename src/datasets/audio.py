import os
import random

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from src.utils.io_utils import ROOT_PATH


def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10 ** (snr / 20)

    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise

    mix = clean + noise_norm

    return mix


def fix_length_(s, L):
    if s.shape[1] >= L:
        s = s[:, :L]
    else:
        s = torch.cat((s, torch.zeros(L - s.shape[1]).to(s.device)[None, :]), dim=1)
    return s


def fix_length(s1, s2, method="max"):
    if method == "min":
        utt_len = min(s1.shape[1], s2.shape[1])
    elif method == "max":
        utt_len = max(s1.shape[1], s2.shape[1])
    else:
        utt_len = method
    s1 = fix_length_(s1, utt_len)
    s2 = fix_length_(s2, utt_len)
    return s1, s2


class GRIDDataset(Dataset):
    """
    GRID dataset

    https://spandh.dcs.shef.ac.uk/avlombard
    """

    def __init__(self, dir, format="wav", lufs=-23.0, transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.dir = dir
        self.files = os.listdir(ROOT_PATH / dir)
        self.format = format
        self.transform = transform
        print("Transform GRIDDataset: ", transform)
        self.lufs = lufs
        self.meter = None

        self.filter_index()

    def filter_index(self):
        self.files = list(filter(lambda name: "WRONG" not in name, self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker = float(self.files[idx].split("_")[0][1:])
        # print(ROOT_PATH / self.dir / self.files[idx])
        signal, sr = torchaudio.load(
            ROOT_PATH / self.dir / self.files[idx], format=self.format
        )

        if not self.meter:
            self.meter = pyln.Meter(sr)

        signal_numpy = signal.numpy()[0, :]
        louds = self.meter.integrated_loudness(signal_numpy)
        signal = torch.Tensor(
            pyln.normalize.loudness(signal_numpy, louds, self.lufs)
        ).unsqueeze(0)

        sample = {"speaker": speaker, "signal": signal, "sr": sr}

        # print("\n\nApply Transform GRIDDataset: ", self.transform, type(self.transform))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class MixedDataset(Dataset):
    def __init__(self, dataset, snr=[0, 10], L=40000, lufs=-23.0, transform=None):
        self.dataset = dataset
        self.transform = transform
        print("Transform MixedDataset: ", transform)
        self.snr = snr
        self.L = L
        self.meter = None
        self.lufs = lufs

    def __len__(self):
        return len(self.dataset) * (len(self.dataset) - 1) // 2

    def __getitem__(self, idx):
        N = len(self.dataset)
        i, j = idx // N, idx % N

        sample_i, sample_j = self.dataset[i], self.dataset[j]
        assert sample_i["sr"] == sample_j["sr"]

        signal1, signal2 = sample_i["signal"], sample_j["signal"]

        signal1, signal2 = fix_length(signal1, signal2, self.L)

        snr = random.randint(*self.snr)
        mixed = snr_mixer(signal1, signal2, snr=snr)

        sample = {
            "speaker1": sample_i["speaker"],
            "speaker2": sample_j["speaker"],
            "signal1": signal1,
            "signal2": signal2,
            "mixed": mixed,
            "sr": sample_i["sr"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
