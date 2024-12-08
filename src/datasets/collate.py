import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["signal_gt"] = torch.vstack(
        [elem["signal_gt"] for elem in dataset_items]
    )
    result_batch["id"] = [elem["id"] for elem in dataset_items]
    result_batch["sr"] = torch.tensor([elem["sr"] for elem in dataset_items])
    result_batch["transcript"] = [elem["transcript"] for elem in dataset_items]
    result_batch["normalized_transcript"] = [elem["normalized_transcript"] for elem in dataset_items]

    # print("Batch:", result_batch)

    return result_batch

def collate_fn_eval(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["id"] = [elem["id"] for elem in dataset_items]
    result_batch["transcript"] = [elem["transcript"] for elem in dataset_items]

    # print("Batch:", result_batch)

    return result_batch

collate_fns = {"train": collate_fn, "val": collate_fn, "eval": collate_fn_eval}
