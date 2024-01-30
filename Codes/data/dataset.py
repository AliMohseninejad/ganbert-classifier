import pandas as pd  
import torch
from typing import *
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random


class GanBertDataset(Dataset):
    """This is the implementation of the semi-supervised Dataset for Gan-Bert
    model
    """

    def __init__(self, tokenizer: BertTokenizer, texts: List[str], labels: List[int], unsupervised_ratio: float, max_length, use_unsup):
        random.seed(42)      
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.use_unsup = use_unsup
        self.is_sup = [0 if random.random() < unsupervised_ratio else 1 for _ in range(len(self.labels))]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the dataset item

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: a tuple of tensors
            in which the first item is the `encoded input`, the second is
            `encoded attention mask` and the third is the `label`. We assume
            that for unsupervised data, the `label` is `-1`.
        """
        text = self.texts[index]
        encoded_input = self.tokenizer(text, padding="max_length", add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'].squeeze(0)
        encoded_attention_mask = self.tokenizer(text, padding="max_length", add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[index])

        # Convert label to one-hot tensor
        #num_classes = len(set(self.labels))
        num_classes = 6
        if self.use_unsup:
            num_classes = num_classes+1
        label_tensor = torch.nn.functional.one_hot(label, num_classes=num_classes).float()
        is_sup_tensor = torch.tensor(self.is_sup[index])

        return encoded_input, encoded_attention_mask, label_tensor, is_sup_tensor


class GanBertBagOfWordsDataset(Dataset):
    """This is the implementation of the semi-supervised Dataset for Gan-Bert
    model with an additional output for the BoW Generative model (BERT)
    """

    def __init__(self, tokenizer: BertTokenizer, texts: List[str], labels: List[int], unsupervised_ratio: float, max_length):
        random.seed(42) 
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.sample_length = 100
        self.words = ' '.join(self.texts).split()
        self.max_length = max_length
        self.is_sup = [0 if random.random() < unsupervised_ratio else 1 for _ in range(len(self.labels))]

    def __len__(self) -> int:
        return len(self.texts)

    def _generate_bag_of_words_sample(self, k) -> str:    
            
        sampled_words = random.sample(self.words, k)
        sampled_text = ' '.join(sampled_words)

        return sampled_text

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the dataset item

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: a
            tuple of tensors in which the first item is the `encoded input`, the
            second is `encoded attention mask`, the third is the `label`, and the
            forth is the `encoded Bag of Words`. We assume that for unsupervised
            data, the `label` is `-1`.
        """
        text = self.texts[index]
        encoded_input = self.tokenizer(text, padding="max_length", add_special_tokens=True,truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'].squeeze(0)
        encoded_attention_mask = self.tokenizer(text, padding="max_length", add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")['attention_mask'].squeeze(0)
        label = torch.tensor(self.labels[index])


        # Convert label to one-hot tensor
        #num_classes = len(set(self.labels))+1
        num_classes = 7
        label_tensor = torch.nn.functional.one_hot(label, num_classes=num_classes).float()

        # Generate Bag of Words sample
        k = 256
        bow_sample_str = self._generate_bag_of_words_sample(k)
        
        # Tokenize BoW sample
        encoded_bow_sample = self.tokenizer(bow_sample_str, padding="max_length", add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'].squeeze(0)
        encoded_bow_attention_mask = self.tokenizer(bow_sample_str, padding="max_length", add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")['attention_mask'].squeeze(0)

        is_sup_tensor = torch.tensor(self.is_sup[index])

        return encoded_input, encoded_attention_mask, label_tensor, encoded_bow_sample, encoded_bow_attention_mask, is_sup_tensor
