import os
import random
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# from src.utils.io_utils import ROOT_PATH


class AudioDataset(Dataset):
    def __init__(self, dir, mode, lufs=-23.0, transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.dir = dir
        self.base = Path(dir) / "audio" / mode
        self.mode = mode
        if mode == "train" or mode == "val":
            self.files = {
                "s1": os.listdir(self.base / "s1"),
                "s2": os.listdir(self.base / "s2"),
                "mix": os.listdir(self.base / "mix"),
            }
        else:
            self.files = {
                "mix": os.listdir(self.base / "mix"),
            }
        self.transform = transform
        print("Transform AudioVideoDataset: ", transform)
        self.lufs = lufs
        self.meter = None

        self.filter_index()

    def filter_index(self):
        self.files = self.files

    def __len__(self):
        return len(self.files["mix"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker1, speaker2 = self.files["mix"][idx].split(".")[0].split("_")
        format = self.files["mix"][idx].split(".")[-1]

        sample = {
            "speaker1": speaker1,
            "speaker2": speaker2,
        }

        if self.mode == "train" or self.mode == "val":
            signal1, sr1 = torchaudio.load(
                self.base / "s1" / self.files["s1"][idx], format=format
            )
            signal2, sr2 = torchaudio.load(
                self.base / "s2" / self.files["s2"][idx], format=format
            )
            mix, sr3 = torchaudio.load(
                self.base / "mix" / self.files["mix"][idx], format=format
            )
            assert sr1 == sr2 == sr3
            sr = sr1

            sample.update(
                {"signal1": signal1, "signal2": signal2, "mixed": mix, "sr": sr}
            )
        else:
            mix, sr3 = torchaudio.load(
                self.base / "mix" / self.files["mix"][idx], format=format
            )
            sr = sr3
            sample.update({"mixed": mix, "sr": sr})

        # if not self.meter:
        #     self.meter = pyln.Meter(sr)
        # signal_numpy = signal.numpy()[0, :]
        # louds = self.meter.integrated_loudness(signal_numpy)
        # signal = torch.Tensor(pyln.normalize.loudness(signal_numpy, louds, self.lufs)).unsqueeze(0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class VideoDataset(Dataset):
    def __init__(self, dir, transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.dir = dir
        self.base = Path(dir) / "mouths"
        self.files = set(os.listdir(self.base))
        self.transform = transform
        print("Transform VideoDataset: ", transform)

        self.filter_index()

    def filter_index(self):
        self.files = self.files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, speaker):
        name = f"{speaker}.npz"

        if name not in self.files:
            print(f"Video for speaker {speaker} not found.")

        video = np.load(self.base / name)["data"]

        video = torch.Tensor(video)

        sample = {"video": video}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class AudioVideoDataset(Dataset):
    def __init__(self, dir, mode, transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.audio = AudioDataset(dir, mode)
        self.mode = mode
        self.video = VideoDataset(dir)
        print("Transform AudioVideoDataset: ", transform)

        self.transform = transform

        self.filter_index()

    def filter_index(self):
        self.audio.filter_index()
        self.video.filter_index()

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, i):
        sample = self.audio[i]
        speaker1 = sample["speaker1"]
        speaker2 = sample["speaker2"]

        v1 = self.video[speaker1]
        v2 = self.video[speaker2]

        sample.update({"video1": v1["video"], "video2": v2["video"]})

        if self.transform is not None:
            for key, transform in self.transform.items():
                sample[key] = transform(sample[key])

        return sample


if __name__ == "__main__":
    import sys

    d = AudioVideoDataset(sys.argv[1], sys.argv[2])
    print(f"len(d) = {len(d)}")
    print("d[1]:")
    for k, v in d[1].items():
        print(f"  {k}: {v}")
