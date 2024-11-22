import warnings

import hydra
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders
from src.metrics.tracker import MetricTracker
from src.trainer import Inferencer
from src.trainer.base_trainer import BaseTrainer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


class MetricsCalculator:
    def __init__(self, config, device, dataloaders, batch_transforms, metrics):
        self.config = config
        self.device = device
        self.dataloaders = dataloaders
        self.batch_transforms = batch_transforms
        self.metrics = metrics

        print("Dataloaders: ", self.dataloaders)
        print("Metrics: ", self.metrics)

        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

    def process_batch(self, batch_idx, batch, metrics):
        batch = self.move_batch_to_device(batch)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        return batch

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.config.inferencer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def calculate(self):
        self.is_train = False

        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.dataloaders["metrics"]),
                total=len(self.dataloaders["metrics"]),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()


@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    """
    Main script for calculating metrics.
    Given the paths to predictions and the ground truths, computes various metrics.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # get metrics
    metrics = instantiate(config.metrics)

    calculator = MetricsCalculator(
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        metrics=metrics,
    )

    logs = calculator.calculate()

    part = "test"
    for key, value in logs.items():
        full_key = part + "_" + key
        print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
