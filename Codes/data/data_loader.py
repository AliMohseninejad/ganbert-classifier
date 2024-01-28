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
    # Set random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Read train, validation, and test data from JSON lines files
    train_df = pd.read_json(dataset_folder_path + "/subtaskB_train.jsonl", lines=True)
    valid_df = pd.read_json(dataset_folder_path + "/subtaskB_dev.jsonl", lines=True)
    test_df = train_df.sample(frac=0.1, random_state=random_seed)  # Use 10% of train data as test data

    # Preprocess data
    def preprocess_data(df):
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return texts, labels

    train_texts, train_labels = preprocess_data(train_df)
    valid_texts, valid_labels = preprocess_data(valid_df)
    test_texts, test_labels = preprocess_data(test_df)

    # Create Dataset objects based on the value of use_bow_dataset
    if use_bow_dataset:
        train_dataset = GanBertBagOfWordsDataset(tokenizer, train_texts, train_labels)
        valid_dataset = GanBertBagOfWordsDataset(tokenizer, valid_texts, valid_labels)
        test_dataset = GanBertBagOfWordsDataset(tokenizer, test_texts, test_labels)
    else:
        train_dataset = GanBertDataset(tokenizer, train_texts, train_labels)
        valid_dataset = GanBertDataset(tokenizer, valid_texts, valid_labels)
        test_dataset = GanBertDataset(tokenizer, test_texts, test_labels)

    # Create DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_dataloader, valid_dataloader, test_dataloader
