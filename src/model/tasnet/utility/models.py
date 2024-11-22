import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Skip connection, dilation
class DepthConv1d(nn.Module):
    def __init__(
        self,
        input_channel,
        hidden_channel,
        kernel,
        padding,
        dilation=1,
    ):
        super(DepthConv1d, self).__init__()

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()

        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        skip = self.skip_out(output)
        return residual, skip


class TCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        BN_dim,
        hidden_dim,
        layer,
        stack,
        kernel=3,
    ):
        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                self.TCN.append(
                    DepthConv1d(
                        BN_dim,
                        hidden_dim,
                        kernel,
                        dilation=2**i,
                        padding=2**i,
                    )
                )

                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    self.receptive_field += (kernel - 1) * 2**i

        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

    def forward(self, input):
        # Input: (B, N, L)

        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        skip_connection = 0.0
        for i in range(len(self.TCN)):
            residual, skip = self.TCN[i](output)
            output = output + residual
            skip_connection = skip_connection + skip

        # output layer
        output = self.output(skip_connection)

        return output
