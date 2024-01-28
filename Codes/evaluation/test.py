from model.discriminator import Discriminator
from model.generator1 import Generator
from typing import *
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertModel
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


def test_vanilla_bert(
    transformer: BertModel,
    generator: Union[Generator, BertModel],
    discriminator: Discriminator,
    test_dataloader: DataLoader,
):
    discriminator.eval()
    transformer.eval()

    test_corrects = 0.0
    test_data_count = 0.0

    test_predictions_f1 = []
    test_true_labels_f1 = []

    with torch.no_grad():
        for batch in test_dataloader:
            # Unpack this training batch from our dataloader.
            encoded_input = batch[0]
            encoded_attention_mask = batch[1]
            labels = batch[2]
            is_supervised = batch[3]

            supervised_indices = torch.nonzero(is_supervised == 1).squeeze()

            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            features, logits, probabilities = discriminator(model_outputs)

            if supervised_indices.shape[0] != 0:
                real_prediction_supervised = probabilities[supervised_indices]
                _, predictions = real_prediction_supervised.max(1)
                correct_predictions_d = (
                    predictions.eq(labels[supervised_indices].max(1)).sum().item()
                )
            else:
                correct_predictions_d = torch.zeros((1,))

            one_hot_predictions = F.one_hot(predictions)
            one_hot_labels = labels[supervised_indices]

            test_data_count += one_hot_labels.size(0)
            test_corrects += correct_predictions_d
            test_predictions_f1.extend(one_hot_predictions.cpu())
            test_true_labels_f1.extend(one_hot_labels.cpu())

        test_accuracy = test_corrects / test_data_count
        test_true_labels_f1_np = np.array(test_true_labels_f1)
        test_predictions_f1_np = np.array(test_predictions_f1)
        test_f1 = f1_score(
            test_true_labels_f1_np, test_predictions_f1_np, average="binary"
        )
    print(f"Test Accuracy equals: {test_accuracy}")
    print(f"Test f1 score equals: {test_f1}")


def test_gan_bert(
    transformer: BertModel,
    generator: Union[Generator, BertModel],
    discriminator: Discriminator,
    bow_mode: bool,
    test_dataloader: DataLoader,
):
    test_corrects = 0.0
    test_data_count = 0.0

    test_predictions_f1 = []
    test_true_labels_f1 = []

    # Validation loop
    generator.eval()
    discriminator.eval()
    transformer.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            # Unpack this training batch from our dataloader.
            encoded_input = batch[0]
            encoded_attention_mask = batch[1]
            labels = batch[2]
            is_supervised = batch[3]
            if bow_mode:
                encoded_bow = batch[4]
                encoded_bow_attention = batch[5]

            supervised_indices = torch.nonzero(is_supervised == 1).squeeze()
            unsupervised_indices = torch.nonzero(is_supervised == 0).squeeze()
            real_batch_size = encoded_input.shape[0]

            # Encode real data in the Transformer
            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            hidden_states = model_outputs[-1]

            # Define noise_size as the same size as the encoded_input
            noise_size = encoded_input.shape[1]

            noise = torch.zeros((real_batch_size, noise_size)).uniform_(0, 1)

            # Train Generator
            if bow_mode:
                generator_outputs = transformer(
                    encoded_bow, attention_mask=encoded_bow_attention
                )
            else:
                generator_outputs = generator(noise)

            # ------------------------------------

            # Train Discriminator
            discriminator_input = torch.cat([generator_outputs, hidden_states], dim=0)

            features, logits, probabilities = discriminator(discriminator_input)

            # Calculate the number of correct predictions for real and fake examples
            fake_predictions = probabilities[:real_batch_size]
            real_predictions = probabilities[real_batch_size:]

            # ------------------------------------------------
            if supervised_indices.shape[0] != 0:
                real_prediction_supervised = real_predictions[supervised_indices]
                sup_fake_probabilities = torch.cat(
                    [fake_predictions, real_prediction_supervised], dim=0
                )
                _, predictions = sup_fake_probabilities.max(1)
                fake_labels = torch.zeros_like(fake_predictions)
                fake_labels[:, -1] = 1
                fake_labels.to(fake_predictions.device)
                all_labels = torch.cat([fake_labels, labels[supervised_indices]], dim=0)
            else:
                _, predictions = fake_predictions.max(1)
                all_labels = torch.zeros_like(fake_predictions)
                all_labels[:, -1] = 1
                all_labels.to(fake_predictions.device)

            _, labels_index = all_labels.max(1)
            correct_predictions_d = predictions.eq(labels_index).sum().item()
            one_hot_predictions = F.one_hot(predictions)
            one_hot_labels = F.one_hot(labels_index)
            # -------------------------------------------------------------------------

            test_data_count += one_hot_labels.size(0)
            test_corrects += correct_predictions_d
            test_predictions_f1.extend(one_hot_predictions.cpu())
            test_true_labels_f1.extend(one_hot_labels.cpu())

    # Calculate average loss and accuracy
    test_accuracy = test_corrects / test_data_count
    test_true_labels_f1_np = np.array(test_true_labels_f1)
    test_predictions_f1_np = np.array(test_predictions_f1)
    test_f1 = f1_score(test_true_labels_f1_np, test_predictions_f1_np, average="binary")

    print(f"Test Accuracy equals: {test_accuracy}")
    print(f"Test f1 score equals: {test_f1}")
