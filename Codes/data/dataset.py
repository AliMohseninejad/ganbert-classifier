import torch
from typing import *
from torch.utils.data import Dataset


class GanBertDataset(Dataset):
    """This is the implementation of the semi-supervised Dataset for Gan-Bert
    model
    """

    def __init__(
        self,
    ):
        pass

    def __len__(
        self,
    ) -> int:
        pass

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the dataset item

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: a tuple of tensors
            in which the first item is the `encoded input`, the second is
            `encoded attention mask` and the third is the `label`. We assume
            that for unsupervised data, the `label` is `-1`.
        """
        pass


class GanBertBagOfWordsDataset(Dataset):
    """This is the implementation of the semi-supervised Dataset for Gan-Bert
    model with an additional output for the BoW Generative model (BERT)
    """

    def __init__(
        self,
    ):
        pass

    def __len__(
        self,
    ) -> int:
        pass

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the dataset item

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: a
            tuple of tensors in which the first item is the `encoded input`, the
            second is `encoded attention mask`, the third is the `label` and the
            forth is the `encoded Bag of Words`.  We assume that for
            unsupervised data, the `label` is `-1`.
        """
        pass
