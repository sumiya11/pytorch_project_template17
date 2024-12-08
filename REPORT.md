# Report

## Introduction

### Model Pipeline



### How to reproduce your model?



### What worked and what didn't work? What were the major challenges?



## Analysis

### Inner-Analysis

> Take several utterances from LJSpeech dataset you were training the vocoder on. 

> Calculate the MelSpectrograms (using original audio) and generate synthesized versions of the audio using your vocoder.

We fetch the training logs from wandb.
Logs contain spectrograms and audios (both ground truth and generated).

- Spectrograms: https://wandb.ai/asdemin_2/pytorch_template/reports/Weave-mel-24-12-08-16-40-10---VmlldzoxMDUxMTYwOQ
- Audio: https://wandb.ai/asdemin_2/pytorch_template/reports/Weave-signal-24-12-08-16-43-23---VmlldzoxMDUxMTY0OA

During training, audios are split in chunks of size 2^13.

> Compare the generated samples with the corresponding original ones. Do this in time and time-frequency domains. What differences do you see? Can you understand that the audio is synthesized by listening to them? Can you do it by looking at the waveform or spectrogram? Explain the results and do some conclusions.

Time-frequency domain: [the table](https://wandb.ai/asdemin_2/pytorch_template/reports/Weave-mel-24-12-08-16-40-10---VmlldzoxMDUxMTYwOQ) has two columns: **mel_gt** (ground truth) and **mel_hat** (generated).

- Visually, generated spectrograms are not as sharp as the ground truth ones. Long horizontal lines on the spectrograms (which correspond to longer-lasting sounds) are blurred on the generated spectrograms.
- The magnitude of some time-frequency regions on the generated spectrograms is sometimes different from the corresponding magnitude on the ground truth spectrograms. Visually, the shade of color is brighter/dimmer.

Time domain: [the table](https://wandb.ai/asdemin_2/pytorch_template/reports/Weave-signal-24-12-08-16-43-23---VmlldzoxMDUxMTY0OA) has two columns: **signal_gt** (ground truth) and **signal_hat** (generated).

- The speech can be recognized from the generated audio, though it contains "robotic" artifacts and noise.  
- The support and the frequencies of the generated signal are usually similar to that of the ground truth signal.
- The magnitudes of the generated signal differ considerably from the ground truth ones.

We suspect that signal data contains physical domain-imposed symmetries and that it is challenging for the Hi-Fi Gan to respect those symmetries.

### External Dataset Analysis

> Take some other utterances, for example, the ones in the [Grade](#grade) Section.

> Calculate the MelSpectrograms and generate synthesized versions.

> Conduct the comparison again. Does the conclusions from the _Inner Analysis_ hold here too? What differences do you see between using external and training datasets?


### Full-TTS system Analysis

> Finally, take some text transcriptions and generate utterances using your vocoder and one of the acoustic models as mentioned in the [Testing](#testing) section. Note that you will need a ground truth audio to do a comparison with:
>   - For the first part, take the text transcription from the LJspeech dataset.
>   - For the second part, take the text transcription from the external dataset.

1

> Conduct the comparison of generated utterances with their original versions. What new artifacts do you see and hear?

2
