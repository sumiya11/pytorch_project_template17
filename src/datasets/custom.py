import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pyloudnorm as pyln
import torchaudio
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas
from librosa.util import normalize

class CustomDirDataset(Dataset):
    def __init__(self, dir, transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.dir = Path(dir)
        self.files = os.listdir(self.dir / 'transcriptions')
        self.filter_index()
        self.transform = transform

        print(f"CustomDirDataset :", len(self), " items")

    def filter_index(self):
        self.files = self.files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        id = self.files[idx].split('.')[0]

        with open(self.dir / 'transcriptions' / f"{id}.txt", 'r') as file:
            transcript = file.readline()

        sample = {
            "id": id,
            'transcript': transcript,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class CustomDirDataset2(Dataset):
    def __init__(self, dir, transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.dir = Path(dir)
        self.wavs = os.listdir(self.dir / 'wavs')
        self.format = 'wav'
        self.transform = transform
        self.filter_index()
        print(f"CustomDirDataset2 :", "\n", self.dir, ",   ", len(self), "items")

    def filter_index(self):
        self.wavs = self.wavs

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        signal, sr = torchaudio.load(
            self.dir / 'wavs' / f"{self.wavs[idx]}", 
            format=self.format
        )

        signal = signal[:, None, :]
        signal = signal[:, :, :(signal.shape[2]//2)]

        id = self.wavs[idx].split('.')[0]

        sample = {
            "id": id,
            "signal_gt": signal, 
            "sr": sr,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
class CustomDirDataset3(Dataset):
    def __init__(self, text, transform=None):
        self.text = text
        self.transform = transform
        print(f"CustomDirDataset3 :", "\n", self.text)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = "cli"

        sample = {
            "id": id,
            'transcript': self.text,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
