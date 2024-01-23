from model.discriminator import Discriminator
from model.generator1 import Generator
from training.loss import GeneratorLossFunction, DiscriminatorLossFunction

import torch
from typing import *
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertModel
from torch.utils.data import DataLoader
import torch.nn as nn


def train_vanilla_classier(
    transformer: BertModel,
    classifier: Discriminator,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    epochs: int,
    scheduler: Union[LRScheduler, None],
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    save_path: str,
) -> Tuple[BertModel, Discriminator, List[Dict[str, float]]]:
    """Main function to train the simple BERT + Discriminator model. In this
    code, the model is trained and validated. In the end the best model is saved
    on disk.

    Args:
        transformer (BertModel): The pre-trained BERT model
        classifier (Discriminator): The Discriminator model
        optimizer (optim.Optimizer): The optimizer
        loss_function (nn.Module): The loss function
        epochs (int): The number of training epochs
        scheduler (Union[LRScheduler, None]): The learning rate scheduler. No
        scheduler is used if the value is `None`.
        train_dataloader (DataLoader): Training data dataloader
        validation_dataloader (DataLoader): Validation data dataloader
        save_path (str): The path to the folder where models should be saved.

    Returns:
        Tuple[BertModel, Discriminator, List[Dict[str, float]]]: A tuple containing
        the trained BERT model, The trained Discriminator and a list of results.
        Each element of the list is a dictionary containing train loss per
        epoch, validation loss per epoch, train accuracy, val accuracy, ...
    """
    pass


def train_gan(
    transformer: BertModel,
    generator: Union[Generator, BertModel],
    discriminator: Discriminator,
    bow_mode: bool,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    generator_loss_function: GeneratorLossFunction,
    discriminator_loss_function: DiscriminatorLossFunction,
    epochs: int,
    generator_scheduler: Union[LRScheduler, None],
    discriminator_scheduler: Union[LRScheduler, None],
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    save_path: str,
) -> Tuple[BertModel, Generator, Discriminator, List[Dict[str, float]]]:
    """Main function to train the BERT-GAN model. In this code, the model is
    trained and validated. In the end the best model is saved on disk.

    Args:
        transformer (BertModel): The pre-trained BERT model
        generator (Generator): The Generator model
        discriminator (Discriminator): The Discriminator model
        bow_mode (bool): If True, The generator is a BERT model. Otherwise it is
        a Generator model.
        generator_optimizer (optim.Optimizer): The Generator optimizer
        discriminator_optimizer (optim.Optimizer): The Discriminator optimizer
        generator_loss_function (GeneratorLossFunction): The generator loss function
        discriminator_loss_function (DiscriminatorLossFunction): The
        discriminator loss function
        epochs (int): number of training epochs
        generator_scheduler (Union[LRScheduler, None]): The Generator learning
        rate scheduler. No scheduler if None.
        discriminator_scheduler (Union[LRScheduler, None]): The Generator
        learning rate scheduler. No scheduler if None.
        train_dataloader (DataLoader): training dataloader
        validation_dataloader (DataLoader): validation dataloader
        save_path (str): The path to the folder where models should be saved.

    Returns:
        Tuple[BertModel, Generator, Discriminator, List[Dict[str, float]]]: A
        tuple containing the trained BERT model, The trained Generator, the
        trained Discriminator and a list of results.  Each element of the list
        is a dictionary containing train loss per epoch, validation loss per
        epoch, train accuracy, val accuracy, ...
    """
    pass
