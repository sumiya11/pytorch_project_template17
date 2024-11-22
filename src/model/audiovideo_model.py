import scipy.signal as sg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import interpolate

from src.model.convtasnet_model import ConvTasNetModel
from src.model.lipreading.model import Lipreading
from src.model.tasnet.utility import models, sdr


# taken from
# https://github.com/vitrioil/Speech-Separation/blob/65a532d36cf0725d622f18ef058cf5a537c01070/src/predict.py#L13
def filter_audio(y, sr=16_000, cutoff=15_000, low_cutoff=1, filter_order=5):
    sos = sg.butter(
        filter_order,
        [low_cutoff / sr / 2, cutoff / sr / 2],
        btype="band",
        analog=False,
        output="sos",
    )
    filtered = sg.sosfilt(sos, y)

    return filtered


class AudioVideoModel(nn.Module):
    def __init__(self, audio, video):
        super(AudioVideoModel, self).__init__()
        self.audio = audio
        self.video = video
        for param in video.parameters():
            param.requires_grad = False
        self.video.eval()

    def forward(self, mixed, video1, video2, **batch):
        # padding
        output, rest = self.audio.pad_signal(mixed)
        batch_size = output.size(0)

        # waveform encoder
        enc_output = self.audio.encoder(output)  # B, N, L
        enc_output_saved = enc_output

        # print("Audio shape: ", enc_output.shape)
        # print(video1.shape)

        video_features1 = self.video(
            video1[:, None, :, :, :], lengths=[video1.shape[0]]
        )
        video_features2 = self.video(
            video2[:, None, :, :, :], lengths=[video2.shape[0]]
        )

        # print(video_features1.shape)

        video_features = torch.cat((video_features1, video_features2), dim=1)
        video_features = torch.transpose(video_features, 2, 1)
        video_features = interpolate(video_features, size=enc_output.shape[-1])

        # print(enc_output.shape)
        # print(video_features.shape)
        enc_output = enc_output + video_features

        # generate masks
        masks = torch.sigmoid(self.audio.TCN(enc_output)).view(
            batch_size, self.audio.num_spk, self.audio.enc_dim, -1
        )  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L

        # print("Masked: ", masked_output.shape)

        # waveform decoder
        output = self.audio.decoder(
            masked_output.view(batch_size * self.audio.num_spk, self.audio.enc_dim, -1)
        )  # B*C, 1, L
        output = output[
            :, :, self.audio.stride : -(rest + self.audio.stride)
        ].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.audio.num_spk, -1)  # B, C, T

        # output = torch.Tensor(filter_audio(output.cpu().detach().numpy())).to(output.device)

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


def test_audiovideomodel():
    x = {
        "mixed": torch.rand(2, 32000),
        "video1": torch.rand(55, 88, 88),
        "video2": torch.rand(55, 88, 88),
    }
    audio = ConvTasNetModel()
    video = Lipreading(
        modality="video",
        hidden_dim=256,
        backbone_type="resnet",
        num_classes=500,
        relu_type="swish",
        tcn_options={
            "num_layers": 4,
            "kernel_size": [3, 5, 7],
            "dropout": 0.2,
            "dwpw": False,
            "width_mult": 1,
        },
        densetcn_options={},
        width_mult=1.0,
        use_boundary=False,
        extract_feats=True,
    )
    nnet = AudioVideoModel(audio, video)
    print(nnet)
    y = nnet(**x)
    print(f"mixed {x['mixed'].shape} -> unmixed {y['unmixed'].shape}")


if __name__ == "__main__":
    test_audiovideomodel()
