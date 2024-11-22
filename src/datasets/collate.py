import torch


def collate_fn(batch: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        batch (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    # print(batch)
    # print(batch[0])

    return {
        "signal1": torch.stack([x["signal1"] for x in batch]),
        "signal2": torch.stack([x["signal2"] for x in batch]),
        "video1": torch.stack([x["video1"] for x in batch]),
        "video2": torch.stack([x["video2"] for x in batch]),
        "mixed": torch.stack([x["mixed"] for x in batch]),
        "speaker1": [x["speaker1"] for x in batch],
        "speaker2": [x["speaker2"] for x in batch],
        "sr": torch.tensor([x["sr"] for x in batch]),
    }
