import torch
from typing import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def generate_dataloader(
    dataset_folder_path: str,
    unsupervised_ratio: float,
    tokenizer: BertTokenizer,
    train_batch_size: int,
    valid_batch_size: int,
    test_batch_size: int,
    use_bow_dataset: bool = False,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the DataLoaders

    Args:
        dataset_folder_path (str): Path to the parent folder of dataset
        unsupervised_ratio (float): A number in range [0..1] that specifies the
        ratio of the trainset that is unsupervised.
        tokenizer (BertTokenizer): The tokenizer Used for encoding the text
        train_batch_size (int): Training data batch-size
        valid_batch_size (int): Validation data batch-size
        test_batch_size (int): Test data batch-size
        user_bow_dataset (bool): if `True`, Use the `GanBertBagOfWordsDataset`
        model. Otherwise use the `GanBertDataset` model.
        random_seed (int, optional): Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing `train
        dataloader`, `validation dataloader` and `test dataloader`.
    """
    pass
