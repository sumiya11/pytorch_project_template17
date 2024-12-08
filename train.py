import warnings

import itertools
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    print("Device:", device)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    generator_params = list(filter(
        lambda p: p.requires_grad, 
        model.generator.parameters()
    ))
    # print("GP", generator_params)
    generator_optimizer = instantiate(config.optimizer, params=generator_params)
    generator_lr_scheduler = instantiate(config.lr_scheduler, optimizer=generator_optimizer)

    discriminator_params = list(filter(
        lambda p: p.requires_grad,
        itertools.chain(model.msd.parameters(), model.mpd.parameters())
    ))
    # print("DP", discriminator_params)

    discriminator_optimizer = instantiate(config.optimizer, params=discriminator_params)
    discriminator_lr_scheduler = instantiate(config.lr_scheduler, optimizer=discriminator_optimizer)


    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    eval_len = config.trainer.get("eval_len")

    trainer = Trainer(
        model=model,
        # criterion=loss_function,
        metrics=metrics,
        optimizer=(generator_optimizer, discriminator_optimizer),
        lr_scheduler=(generator_lr_scheduler, discriminator_lr_scheduler),
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        eval_len=eval_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", False),
    )

    trainer.train()


if __name__ == "__main__":
    main()
