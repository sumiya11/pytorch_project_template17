# Report

## Introduction

### Model Pipeline

We implement the HiFiGan (version 2) described in https://arxiv.org/abs/2010.05646.
There also exists a reference implementation: https://github.com/jik876/hifi-gan.
 
We tried to implement the description from the paper as closely as possible.
The description of the model in the paper is under-specified, 
so we consulted the reference implementation to get the values of some parameters.

We chose to implement version 2 because it is the smallest one and we do not have a powerful GPU.

Just as in the original paper, the audios are split in small segments, 
MEL spectrograms are calculated and fed into the GAN to produce an audio.

For text-to-speech synthesis we use a pretrained FastPitch.

### How to reproduce your model?

We spent 551 test runs to train the model. The run for the final model is: https://wandb.ai/asdemin_2/pytorch_template/runs/uqofyzkd.

Our dataset is LJSpeech. We used 200 utterances for validation: `data/val.csv`.

For training, we used the command:

```
python train.py --config-name train
```

Training took around 2 days on a GPU with 10 GB memory.

The final model is available at https://drive.google.com/file/d/1LK5kyJ4i5UD36BfVfU9BDtUsywld4oSd/view?usp=sharing.

### What worked and what didn't work? What were the major challenges?

At first, we tried (for no particular reason) to train the text-to-speech pipeline.
That is, we produced MEL spectrograms from texts using a pretrained FastPitch and fed those spectrograms into HiFi GAN to train it. This did not work (generator loss was nearly constant). We concluded that the reason was that we did not align the data reasonably: text vs. spectrogram vs. audio. After, we switched to speech-to-speech training.

We tried different scheduling algorithms, slightly different MEL settings, and different audio segment sizes, this did not make a big change in the quality of generation.
We note that training with large segment size requires a lot of GPU memory (apparently because of the costly MEL representation).

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

We take the utterances from the DLA telegram channel and put them in `data/wavs/`.

> Calculate the MelSpectrograms and generate synthesized versions.

We run our HiFi GAN using the **speech-to-speech** command from the README.md and put the generated wavs in `data/examples/example2/`.

> Conduct the comparison again. Does the conclusions from the _Inner Analysis_ hold here too? What differences do you see between using external and training datasets?

See `supplementary.ipynb`, Part 1.

### Full-TTS system Analysis

> Finally, take some text transcriptions and generate utterances using your vocoder and one of the acoustic models as mentioned in the [Testing](#testing) section. Note that you will need a ground truth audio to do a comparison with:
>   - For the first part, take the text transcription from the LJspeech dataset.
>   - For the second part, take the text transcription from the external dataset.

We take the texts from the DLA telegram channel and put them in `data/transcriptions/`.

We run FastPitch + HiFi GAN using the **text-to-speech** command from the README.md and put the generated wavs in `data/examples/example/`.

> Conduct the comparison of generated utterances with their original versions. What new artifacts do you see and hear?

See `supplementary.ipynb`, Part 2.
