import torch
from torch import nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.functional.audio import permutation_invariant_training

from src.model.tasnet.utility.sdr import batch_SDR_torch, calc_sdr_torch


class MseLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.MSELoss(**kwargs)

    def forward(
        self,
        unmixed: torch.Tensor,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        **batch
    ):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        y_batch = torch.stack((signal1, signal2), dim=1).squeeze(2)

        loss = self.loss(unmixed, y_batch)

        return {"loss": loss}


class PIT_Loss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, metric_func, mode, eval_func, **kwargs):
        super().__init__()
        self.metric_func = metric_func
        self.mode = mode
        self.eval_func = eval_func

    def forward(
        self,
        unmixed: torch.Tensor,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        **batch
    ):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        y_batch = torch.stack((signal1, signal2), dim=1).squeeze(2)

        # loss = ScaleInvariantSignalNoiseRatio().to(unmixed.device)(unmixed[:, :1, :], y_batch[:, :1, :])
        # loss = -loss.mean()

        loss = nn.MSELoss(reduction="sum")(unmixed, y_batch)

        # loss = loss - batch_SDR_torch(unmixed[:, :1, :], y_batch[:, :1, :]).mean()
        # loss = -loss.mean()

        # loss, perm = permutation_invariant_training(unmixed, y_batch,
        #     ScaleInvariantSignalNoiseRatio().to(unmixed.device), mode=self.mode, eval_func=self.eval_func)
        # loss = -loss.sum()

        return {"loss": loss}
