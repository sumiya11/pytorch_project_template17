import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

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
