import torch

from src.metrics.base_metric import BaseMetric
from src.model.tasnet.utility.sdr import batch_SDR_torch


class SDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        unmixed: torch.Tensor,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        **kwargs
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        y_batch = torch.stack((signal1, signal2), dim=1).squeeze(2)
        return batch_SDR_torch(unmixed[:, 0:1, :], y_batch[:, 0:1, :]).sum()


class SomeMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(
        self,
        unmixed: torch.Tensor,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        **kwargs
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        y_batch = torch.stack((signal1, signal2), dim=1).squeeze(2)

        return self.metric(unmixed[:, 0:1, :], y_batch[:, 0:1, :])


# class PESQMetric(BaseMetric):
#     def __init__(self, metric, device, *args, **kwargs):
#         """
#         Example of a nested metric class. Applies metric function
#         object (for example, from TorchMetrics) on tensors.

#         Notice that you can define your own metric calculation functions
#         inside the '__call__' method.

#         Args:
#             metric (Callable): function to calculate metrics.
#             device (str): device for the metric calculation (and tensors).
#         """
#         super().__init__(*args, **kwargs)
#         if device == "auto":
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.metric = metric.to(device)

#     def __call__(
#         self,
#         unmixed: torch.Tensor,
#         signal1: torch.Tensor,
#         signal2: torch.Tensor,
#         **kwargs
#     ):
#         """
#         Metric calculation logic.

#         Args:
#             logits (Tensor): model output predictions.
#             labels (Tensor): ground-truth labels.
#         Returns:
#             metric (float): calculated metric.
#         """

#         y_batch = torch.stack((signal1, signal2), dim=1).squeeze(2)

#         return self.metric(unmixed, y_batch)
