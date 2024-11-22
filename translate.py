import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

path = Path(sys.argv[1])
dest = Path(sys.argv[2])

os.makedirs(dest / "s1", exist_ok=True)
os.makedirs(dest / "s2", exist_ok=True)

sr = 16000


def rename(file):
    file = ".".join(file.split(".")[:-1]) + ".wav"
    return "_".join(file.split("_")[1:])


for file in tqdm(os.listdir(path), desc="Postprocess, .pth -> .wav"):
    unmixed = torch.load(path / file)
    unmixed = unmixed["unmixed"].cpu()
    torchaudio.save(dest / "s1" / rename(file), unmixed[0, 0:1, :], sr)
    torchaudio.save(dest / "s2" / rename(file), unmixed[0, 1:2, :], sr)
