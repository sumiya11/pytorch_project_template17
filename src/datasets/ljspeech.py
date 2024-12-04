import os
import random

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas

from src.model.hifi_gan import HiFiConfig

class LJSpeech(Dataset):
    def __init__(self, dir, hifi_config, mode='train', transform=None):
        """
        Arguments:
            dir (string): A path, root directory.
        """
        self.mode = mode
        self.hifi_config = hifi_config
        self.dir = Path(dir)
        self.wavs = os.listdir(self.dir / 'wavs')
        self.metadata = pandas.read_table(
            self.dir / 'metadata.csv', 
            header=None,
            delimiter='|')
        print(self.metadata)
        self.id_to_transcript = {
            self.metadata.loc[i, 0] : (self.metadata.loc[i, 1], self.metadata.loc[i, 2])
            for i in range(len(self.metadata))    
        }
        print(self.id_to_transcript[self.metadata.loc[0, 0]])
        self.format = 'wav'
        self.transform = transform
        print("Transform LJSpeech: ", transform)
        self.filter_index()

    def filter_index(self):
        self.wavs = self.wavs

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        signal, sr = torchaudio.load(
            self.dir / 'wavs' / self.wavs[idx], 
            format=self.format
        )

        signal = signal / self.hifi_config.MAX_WAV_VALUE

        id = self.wavs[idx].split('.')[0]

        transcript, normalized_transcript = self.id_to_transcript[id]

        if signal.shape[1] >= self.hifi_config.segment_size:
            max_audio_start = signal.shape[1] - self.hifi_config.segment_size
            audio_start = random.randint(0, max_audio_start)
            signal = signal[:, audio_start:audio_start + self.hifi_config.segment_size]
        else:
            signal = torch.nn.functional.pad(signal, (0, self.hifi_config.segment_size - signal.shape[1]), 'constant')

        signal = signal[:, None, :]

        sample = {
            "id": id,
            "signal_gt": signal, 
            "sr": sr,
            'transcript': transcript,
            'normalized_transcript': normalized_transcript,
        }

        # print("\n\nApply Transform GRIDDataset: ", self.transform, type(self.transform))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    
if __name__ == "__main__":
    d = LJSpeech(
        r"C:\Downloads_2\LJSpeech-1.1",
        HiFiConfig()
    )
    print(f"len(d) = {len(d)}")
    print("d[1]:")
    for k, v in d[1].items():
        print(f"  {k}: {v}")
    print(f'signal.shape:  {d[1]['signal'].shape}')
