# adapted from https://github.com/naplab/Conv-TasNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model.tasnet.utility import models, sdr


class ConvTasNetModel(nn.Module):
    def __init__(
        self,
        enc_dim=512,
        feature_dim=128,
        sr=16000,
        win=2,
        layer=8,
        stack=3,
        kernel=3,
        num_spk=2,
    ):
        super(ConvTasNetModel, self).__init__()

        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim

        self.win = int(sr * win / 1000)
        self.stride = self.win // 2

        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        # input encoder
        self.encoder = nn.Conv1d(
            1, self.enc_dim, self.win, bias=False, stride=self.stride
        )

        # TCN separator
        self.TCN = models.TCN(
            self.enc_dim,
            self.enc_dim * self.num_spk,
            self.feature_dim,
            self.feature_dim * 4,
            self.layer,
            self.stack,
            self.kernel,
        )

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(
            self.enc_dim, 1, self.win, bias=False, stride=self.stride
        )

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, mixed, **batch):
        # padding
        output, rest = self.pad_signal(mixed)
        batch_size = output.size(0)

        # print(output.shape)

        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L

        # print(enc_output.shape)

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(
            batch_size, self.num_spk, self.enc_dim, -1
        )  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L

        # print(masked_output.shape)

        # waveform decoder
        output = self.decoder(
            masked_output.view(batch_size * self.num_spk, self.enc_dim, -1)
        )  # B*C, 1, L
        output = output[
            :, :, self.stride : -(rest + self.stride)
        ].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return {"unmixed": output}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = ConvTasNetModel()
    print(nnet)
    y = nnet(x)
    print(f"{x.shape} -> {y.shape}")


if __name__ == "__main__":
    test_conv_tasnet()
