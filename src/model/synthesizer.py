import torch
from torch import nn
import contextlib
import io
import sys

from dataclasses import dataclass

from src.model.hifi_gan import HiFiGAN, model_report, HiFiConfig, MelSpectrogramConfig

# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

class Synthesizer(nn.Module):
    def __init__(
            self,
            hifi_cfg, mel_cfg, 
            pace=1.0,
            speaker=0,
            cmudict_path="cmudict-0.7b", heteronyms_path="heteronyms"):
        super().__init__()

        self.pace = pace
        self.speaker = speaker
        
        self.text_processor = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 
            'nvidia_textprocessing_utils', 
            cmudict_path=cmudict_path, 
            heteronyms_path=heteronyms_path
        )

        hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
        self._hifigan = hifigan
        self._denoiser = denoiser
        self._vocoder_train_setup = vocoder_train_setup
    
        generator, generator_train_setup = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_fastpitch'
        )
        self.generator = generator
        self.generator.eval()
        self.gen_kw = {
            'pace': self.pace,
            'speaker': self.speaker,
            'pitch_tgt': None,
            'pitch_transform': None
        }

        CHECKPOINT_SPECIFIC_ARGS = [
            'sampling_rate', 'hop_length', 'win_length', 'p_arpabet', 'text_cleaners',
            'symbol_set', 'max_wav_value', 'prepend_space_to_text',
            'append_space_to_text']

        for k in CHECKPOINT_SPECIFIC_ARGS:
            v1 = generator_train_setup.get(k, None)
            v2 = vocoder_train_setup.get(k, None)
            assert v1 is None or v2 is None or v1 == v2, \
                f'{k} mismatch in spectrogram generator and vocoder'
        

        for param in self.generator.parameters():
            param.requires_grad = False

        for param in self._hifigan.parameters():
            param.requires_grad = False

        for param in self._denoiser.parameters():
            param.requires_grad = False

        self.vocoder = HiFiGAN(hifi_cfg, mel_cfg)

        # TODO: check parameters are okay


    def forward(self, batch):
        # transcript = batch['transcript']

        # with nostdout():
        #     text_processed = self.text_processor.prepare_input_sequence(
        #         transcript, batch_size=len(transcript))

        # self.generator.eval()
        # device = next(self.generator.parameters()).device
        # mel, mel_lens, *_ = self.generator(
        #     text_processed[0]['text'].to(device), **self.gen_kw)

        # mel = mel[:, :, :mel_gt.shape[2]]
        # batch.update({'mel_gt': mel_gt})

        result = {}
        output = self.vocoder(batch)
        result.update(output)

        # with torch.no_grad():
        #     denoising_strength = 0.005
        #     reference = self._hifigan(mel).float()
        #     reference = self._denoiser(reference.squeeze(1), denoising_strength)
        #     reference = reference # * self._vocoder_train_setup['max_wav_value']
        #     batch.update({'reference': reference})

        return result
    
    def losses(self, batch):
        losses = self.vocoder.losses(batch)
        return losses

    def __str__(self):
        result_info = super().__str__()
        result_info = result_info + "\nText processor\n" + repr(self.text_processor)
        result_info = result_info + "\nGenerator" + model_report(self.generator)
        result_info = result_info + "\nVocoder (HiFi)" + model_report(self.vocoder)
        result_info = result_info + "\nTotal" + model_report(self)
        return result_info
    
if __name__ == "__main__":
    import os
    
    text = [
        'Hello world!',
        'Goodbye world!'
    ]

    hifi_cfg = HiFiConfig()
    mel_cfg = MelSpectrogramConfig()

    print(os.path.abspath("heteronyms"))

    text_processor = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 
            'nvidia_textprocessing_utils', 
            cmudict_path="cmudict-0.7b", 
            heteronyms_path="heteronyms"
        )
    print(text_processor.prepare_input_sequence(text, batch_size=2))

    synth = Synthesizer(
        hifi_cfg, 
        mel_cfg,
        cmudict_path="cmudict-0.7b", 
        heteronyms_path="heteronyms"
    )
    print(synth)

    synth.train()

    batch = { 
        'transcript': text, 
        'signal_gt': torch.randn(2, 1, 20000)
    }
    
    batch = synth(batch)

    print('Forward')
    for k, v in batch.items():
        try:
            print(f"  {k}: {v.shape}")
        except:
            print(f"  {k}: {type(v), len(v)}")

    batch = synth.losses(batch)

    print(f"losses")
    for k, v in batch.items():
        try:
            print(f"  {k}: {v.shape}")
        except:
            print(f"  {k}: {type(v), len(v)}")
