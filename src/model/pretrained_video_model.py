import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model.lipreading.model import Lipreading
from src.model.lipreading.utils import (
    AverageMeter,
    CheckpointSaver,
    calculateNorm2,
    get_logger,
    get_save_folder,
    load_json,
    load_model,
    save2npz,
    showLR,
    update_logger_batch,
)

# from src.model.lipreading.mixup import mixup_data, mixup_criterion
# from src.model.lipreading.optim_utils import get_optimizer, CosineScheduler
# from src.model.lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines


class PretrainedVideoModel(nn.Module):
    def __init__(self, path, model):
        super(PretrainedVideoModel, self).__init__()
        self.model = model
        self.path = path
        self.model = load_model(path, self.model, allow_size_mismatch=False)
        self.model.eval()

    def forward(self, *args, **kwargs):
        self.model.eval()
        return self.model(*args, **kwargs)

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


def test_lipreading():
    import sys

    x = torch.rand(50, 88, 88)
    nnet = Lipreading(
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
    nnet = load_model(sys.argv[1], nnet, allow_size_mismatch=False)
    nnet = nnet
    print(f"Model has been successfully loaded from {sys.argv[1]}")
    print(nnet)
    nnet.eval()
    y = nnet(x[None, None, :, :, :], lengths=[x.shape[0]])
    print(f"{x.shape} -> {y.shape}")


if __name__ == "__main__":
    test_lipreading()
