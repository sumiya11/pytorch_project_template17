"""
HiFiGan implementation.

Closely follows paper
    
    https://arxiv.org/abs/2010.05646.

Direct quotes from the paper are given by

    > Quote.

"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Conv1d, Conv2d, AvgPool1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from dataclasses import dataclass
import torchaudio
import librosa

from librosa.filters import mel as librosa_mel_fn

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

'''
> where we set λfm = 2 and λmel = 45.
'''
Lambda_Feature_Loss = 2
Lambda_mel_loss = 45

PRINT_VERSIONS = True

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1025
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    fmin = 0
    fmax = 8000

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251

@dataclass
class HiFiConfig:
    num_mels = 80
    num_freq = 1025
    upsample_rates = [8,8,2,2]
    upsample_kernel_sizes = [16,16,4,4]
    upsample_initial_channel = 128
    resblock_kernel_sizes = [3,7,11]
    resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
    leaky_relu_slope = 0.1

    segment_size: int = 8192
    MAX_WAV_VALUE = 1.0

    
class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb = torch.tensor(mel_basis)

        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        print("mel forward: ", audio.shape)

        # audio = torch.nn.functional.pad(audio.unsqueeze(1), (int((self.config.n_fft - self.config.hop_length)/2), int((self.config.n_fft - self.config.hop_length)/2)), mode='reflect')
        # audio = audio.squeeze(1)

        mel = torch.log(torch.clamp(self.mel_spectrogram(audio), min=1e-5))

        return mel

        # if self.config.fmax not in self.mel_basis:
        #     mel = librosa.filters.mel(
        #         sr=self.config.sr, 
        #         n_fft=self.config.n_fft, 
        #         n_mels=self.config.n_mels, 
        #         fmin=self.config.fmin, 
        #         fmax=self.config.fmax
        #     )
        #     self.mel_basis[str(self.config.fmax) + '_' + str(audio.device)] = torch.from_numpy(mel).float().to(audio.device)
        #     self.hann_window[str(audio.device)] = torch.hann_window(self.config.win_length).to(audio.device)

        # audio = torch.nn.functional.pad(audio, (int((self.config.n_fft - self.config.hop_length)/2), int((self.config.n_fft - self.config.hop_length)/2)), mode='reflect')
        # # audio = audio.squeeze(1)

        # spec = torch.stft(audio, self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length, window=self.hann_window[str(audio.device)],
        #                 center=False, pad_mode='reflect', normalized=False, onesided=True,
        #                 return_complex=False)

        # spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        # spec = torch.matmul(self.mel_basis[str(self.config.fmax)+'_'+str(audio.device)], spec)
        # spec = dynamic_range_compression_torch(spec)

        # return spec

'''
> Feature Matching Loss
'''
def feature_loss(hat_feat, gt_feat):
    loss = torch.zeros((1,)).to(hat_feat[0][0].device)
    for D, DG in zip(hat_feat, gt_feat):
        for Di, DGi in zip(D, DG):
            loss = loss + torch.mean(torch.abs(Di - DGi))
    return loss

# GAN Loss
def discriminator_loss(gt_disc, hat_disc):
    loss = torch.zeros((1,)).to(gt_disc[0].device)
    for dr, dg in zip(gt_disc, hat_disc):
        gt_loss = torch.mean((dr - 1)**2)
        hat_loss = torch.mean(dg**2)
        loss = loss + (gt_loss + hat_loss)
    return loss

# GAN Loss
def generator_loss(disc_outputs):
    loss = torch.zeros((1,)).to(disc_outputs[0].device)
    for D in disc_outputs:
        l = torch.mean((D - 1)**2)
        loss = loss + l
    return loss

def model_report(model, indent='  '):
    all_parameters = sum([p.numel() for p in model.parameters()])
    trainable_parameters = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    result_info = ""
    result_info = result_info + f"\n{indent}All parameters: {all_parameters}"
    result_info = result_info + f"\n{indent}Trainable parameters: {trainable_parameters}"
    return result_info

def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, Conv1d) or isinstance(m, ConvTranspose1d):
        m.weight.data.normal_(mean, std)
    elif isinstance(m, ModuleList):
        for elem in m:
            init_weights(elem, mean=mean, std=std)
    else:
        raise RuntimeError(f"Unknown module: {m}")

def padding(kernel, dilation):
        return (kernel * dilation - dilation) // 2

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        self.conv1 = nn.ModuleList()
        for dilate in dilation:
            conv = Conv1d(
                in_channels=channels, 
                out_channels=channels, 
                kernel_size=kernel_size, 
                stride=1, 
                dilation=dilate,
                padding=padding(kernel_size, dilate)
            )
            self.conv1.append(conv)

        self.conv2 = nn.ModuleList()
        for _ in range(3):
            dilate = 1
            conv = Conv1d(
                in_channels=channels, 
                out_channels=channels, 
                kernel_size=kernel_size, 
                stride=1, 
                dilation=dilate,
                padding=padding(kernel_size, dilate)
            )
            self.conv2.append(conv)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.conv1, self.conv2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, hifi_cfg):
        super().__init__()

        self.hifi_cfg = hifi_cfg

        self.n_upsamples = len(self.hifi_cfg.upsample_rates)
        self.n_kernels = len(self.hifi_cfg.resblock_kernel_sizes)

        self.conv_pre = weight_norm(Conv1d(self.hifi_cfg.num_mels, self.hifi_cfg.upsample_initial_channel, 7, 1, padding=3))

        self.upsample = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(self.hifi_cfg.upsample_rates, self.hifi_cfg.upsample_kernel_sizes)):
            conv = ConvTranspose1d(
                in_channels=self.hifi_cfg.upsample_initial_channel // (2**i), 
                out_channels=self.hifi_cfg.upsample_initial_channel // (2**(i+1)),
                kernel_size=kernel, 
                stride=rate, 
                padding=(kernel - rate) // 2
            )
            self.upsample.append(conv)

        self.resblocks = nn.ModuleList()
        for i in range(self.n_upsamples):
            channels = self.hifi_cfg.upsample_initial_channel // (2**(i+1))
            for _, (kernel, dilation) in enumerate(zip(self.hifi_cfg.resblock_kernel_sizes, self.hifi_cfg.resblock_dilation_sizes)):
                resblock = ResBlock(
                    channels, 
                    kernel_size=kernel, 
                    dilation=dilation, 
                    leaky_relu_slope=self.hifi_cfg.leaky_relu_slope
                )
                self.resblocks.append(resblock)

        self.conv_post = weight_norm(Conv1d(channels, 1, 7, 1, padding=3))

        self.upsample.apply(init_weights)
        self.conv_pre.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        in: [B, C, M], spectrogram
        out: [B, 1, N], signal
        """
        x = self.conv_pre(x)
        for i in range(self.n_upsamples):
            x = F.leaky_relu(x, self.hifi_cfg.leaky_relu_slope)
            x = self.upsample[i](x)
            xs = self.resblocks[i*self.n_kernels](x)
            for j in range(1, self.n_kernels):
                xs = xs + self.resblocks[i*self.n_kernels + j](x)
            x = xs / self.n_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

class DiscriminatorP(nn.Module):
    def __init__(self, 
            period, kernel_size=5, stride=3, leaky_relu_slope=0.1,
            norm=weight_norm):
        super().__init__()
        
        self.period = period
        self.leaky_relu_slope = leaky_relu_slope

        '''
        Note: parameters of this Discriminator are not specified in the paper.
        Using parameters from the reference implementation.
        '''
        self.convs = nn.ModuleList([
            norm(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(padding(5, 1), 0))),
            norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(padding(5, 1), 0))),
            norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(padding(5, 1), 0))),
            # norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(padding(5, 1), 0))),
            # norm(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm(Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        '''
        > We first reshape 1D raw audio of length T into 2D data of height T /p
          and width p and then apply 2D convolutions to the reshaped data. 
        '''
        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period, self.period)

        '''
        > Each sub-discriminator is a stack of strided convolutional layers
        with leaky rectified linear unit (ReLU) activation.
        '''
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, self.leaky_relu_slope)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat
    
class MultiPeriodDiscriminator(torch.nn.Module):

    '''
    > We set the periods to [2, 3, 5, 7, 11] to avoid overlaps as much as possible.
    '''
    def __init__(self, periods=[2, 3, 5, 7, 11], leaky_relu_slope=0.1):
        super().__init__()
        
        self.periods = periods
        self.leaky_relu_slope = leaky_relu_slope

        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(
                DiscriminatorP(period, leaky_relu_slope=leaky_relu_slope))

    def forward(self, signal_gt, signal_hat):
        signal_gt_discr = []
        signal_hat_discr = []
        signal_gt_feat = []
        signal_hat_feat = []
        for discr in self.discriminators:
            gt_discr, gt_feat = discr(signal_gt)
            hat_discr, hat_feat = discr(signal_hat)
            signal_gt_discr.append(gt_discr)
            signal_hat_discr.append(hat_discr)
            signal_gt_feat.append(gt_feat)
            signal_hat_feat.append(hat_feat)
        return {
            'gt_discr_mpd' : signal_gt_discr, 
            'hat_discr_mpd': signal_hat_discr,
            'gt_feat_mpd' : signal_gt_feat, 
            'hat_feat_mpd': signal_hat_feat
        }
    
class DiscriminatorS(nn.Module):
    def __init__(self, leaky_relu_slope=0.1, norm=weight_norm):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        '''
        Note: parameters of this Discriminator are not specified in the paper.
        Using parameters from the reference implementation.
        '''

        self.convs = nn.ModuleList([
            norm(Conv1d(1, 128, 15, 1, padding=7)),
            norm(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            # norm(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            # norm(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            # norm(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(Conv1d(512, 1, 3, 1, padding=1))

    def forward(self, signal):
        # TODO: shapes
        feat = []
        for conv in self.convs:
            signal = conv(signal)
            signal = F.leaky_relu(signal, self.leaky_relu_slope)
            feat.append(signal)
        signal = self.conv_post(signal)
        feat.append(signal)
        signal = torch.flatten(signal, 1, -1)
        return signal, feat

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, leaky_relu_slope=0.1):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        '''
        > Raw audio, x2 average-pooled audio, and x4 average-pooled audio.
        > Weight normalization is applied except for the first sub-discriminator.
        > Spectral normalization is applied for the first sub-discriminator.
        '''

        discr1 = nn.Sequential(
            DiscriminatorS(leaky_relu_slope=self.leaky_relu_slope, 
                            norm=spectral_norm),
        )

        discr2 = nn.Sequential(
            AvgPool1d(4, 2, padding=2),
            DiscriminatorS(leaky_relu_slope=self.leaky_relu_slope),
        )

        discr3 = nn.Sequential(
            AvgPool1d(4, 2, padding=2),
            DiscriminatorS(leaky_relu_slope=self.leaky_relu_slope),
        )

        self.discriminators = nn.ModuleList([discr1, discr2, discr3])

    def forward(self, signal_gt, signal_hat):
        # TODO: shapes
        signal_gt_discr = []
        signal_hat_discr = []
        signal_gt_feat = []
        signal_hat_feat = []
        for discr in self.discriminators:
            gt_discr, gt_feat = discr(signal_gt)
            hat_discr, hat_feat = discr(signal_hat)
            signal_gt_discr.append(gt_discr)
            signal_hat_discr.append(hat_discr)
            signal_gt_feat.append(gt_feat)
            signal_hat_feat.append(hat_feat)
        return {
            'gt_discr_msd' : signal_gt_discr, 
            'hat_discr_msd': signal_hat_discr,
            'gt_feat_msd' : signal_gt_feat, 
            'hat_feat_msd': signal_hat_feat
        }
    
class HiFiGAN(nn.Module):
    """
    HiFi-GAN v2 (smaller one).

    https://arxiv.org/abs/2010.05646
    """

    def __init__(self, hifi_cfg, mel_cfg):
        super().__init__()

        self.mel_cfg = mel_cfg
        self.hifi_cfg = hifi_cfg
        self.mel = MelSpectrogram(mel_cfg)

        self.generator = Generator(hifi_cfg)

        self.mpd = MultiPeriodDiscriminator()

        self.msd = MultiScaleDiscriminator()

    def forward(self, batch):
        # TODO: shapes
        
        result = {}

        mel_gt = self.mel(batch['signal_gt'].squeeze(1))
        result.update({'mel_gt': mel_gt})

        signal_hat = self.generator(mel_gt)
        result.update({'signal_hat' : signal_hat})

        if not 'signal_gt' in batch:
            return result

        signal_gt = batch['signal_gt']
        result.update({'signal_hat' : signal_hat})

        mel_hat = self.mel(signal_hat.squeeze(1))
        result.update({'mel_hat' : mel_hat})
        
        discr_mpd = self.mpd(signal_gt, signal_hat)
        discr_msd = self.msd(signal_gt, signal_hat)

        result.update(discr_mpd)
        result.update(discr_msd)

        if PRINT_VERSIONS:
            print ('versions:')
            print ('  mel_gt:', mel_gt._version)
            print ('  signal_hat:', signal_hat._version)
            print ('  mel_hat:', mel_hat._version)

        return result
    
    def losses(self, batch):
        result = {}

        mel_gt = batch['mel_gt']
        mel_hat = batch['mel_hat']

        hat_discr_mpd = batch['hat_discr_mpd']
        gt_discr_mpd = batch['gt_discr_mpd']
        hat_feat_mpd = batch['hat_feat_mpd']
        gt_feat_mpd = batch['gt_feat_mpd']
        hat_discr_msd = batch['hat_discr_msd']
        gt_discr_msd = batch['gt_discr_msd']
        hat_feat_msd = batch['hat_feat_msd']
        gt_feat_msd = batch['gt_feat_msd']
        
        loss_mel = F.l1_loss(mel_gt, mel_hat)
        loss_feat_mpd = feature_loss(hat_feat_mpd, gt_feat_mpd)
        loss_feat_msd = feature_loss(hat_feat_msd, gt_feat_msd)
        loss_gen_mpd = generator_loss(hat_discr_mpd)
        loss_gen_msd = generator_loss(hat_discr_msd)
        loss_gen = loss_gen_mpd + loss_gen_msd + Lambda_Feature_Loss * (loss_feat_mpd + loss_feat_msd) + Lambda_mel_loss * loss_mel

        loss_disc_mpd = discriminator_loss(gt_discr_mpd, hat_discr_mpd)
        loss_disc_msd = discriminator_loss(gt_discr_msd, hat_discr_msd)

        loss_disc = loss_disc_mpd + loss_disc_msd

        if PRINT_VERSIONS:
            print ('versions:')
            print ('  loss_mel:', loss_mel._version)
            print ('  loss_gen:', loss_gen._version)
            print ('  loss_disc:', loss_disc._version)
              
        result.update({
            'loss_disc': loss_disc, 
            'loss_gen': loss_gen
        })

        return result

    def __str__(self):
        result_info = super().__str__()
        result_info = result_info + "\nGenerator" + model_report(self.generator)
        result_info = result_info + "\nMSD" + model_report(self.msd)
        result_info = result_info + "\nMPD" + model_report(self.mpd)
        result_info = result_info + "\nTotal" + model_report(self)
        return result_info
    
if __name__ == "__main__":
    hifi_cfg = HiFiConfig()
    mel_cfg = MelSpectrogramConfig()

    hifi = HiFiGAN(hifi_cfg, mel_cfg)
    print(hifi)

    hifi.train()

    signal_gt = torch.randn(4, 1, 20000)
    mel = hifi.mel(signal_gt.squeeze(1))
    print("signal_gt", signal_gt.shape)
    print("mel", mel.shape)
    batch = {'mel_gt': mel, 'signal_gt': signal_gt}
    
    y = hifi(**batch)

    print(f"mel: {mel.shape}, signal_gt: {signal_gt.shape}  ->")
    for k, v in y.items():
        try:
            print(f"  {k}: {v.shape}")
        except:
            print(f"  {k}: {type(v), len(v)}")

    y = hifi.losses(**y)

    print(f"signal_gt: {signal_gt.shape}  ->")
    for k, v in y.items():
        try:
            print(f"  {k}: {v.shape}")
        except:
            print(f"  {k}: {type(v), len(v)}")
