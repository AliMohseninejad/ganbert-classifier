from model.bert import get_bert_model
from model.discriminator import Discriminator
from model.generator1 import Generator1
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
import random


def test_vanilla_bert(
    bert_model_name: str,
    transformer_path: str,
    discriminator_path: str,
    test_dataloader: DataLoader,
):
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer, _ = get_bert_model(bert_model_name)
    transformer.load_state_dict(torch.load(transformer_path))
    transformer.to(device)
    transformer.eval()
    discriminator = Discriminator(num_labels=6)
    discriminator.load_state_dict(torch.load(discriminator_path))
    discriminator.to(device)
    discriminator.eval()

    test_corrects = 0.0
    test_data_count = 0.0

    test_predictions_f1 = []
    test_true_labels_f1 = []
    for batch in test_dataloader:
        # Unpack this training batch from our dataloader.
        encoded_input = batch[0].to(device)
        encoded_attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        is_supervised = batch[3].to(device)

        with torch.no_grad():
            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            features, logits, probabilities = discriminator(model_outputs[-1])

            real_prediction_supervised = probabilities
            _, predictions = real_prediction_supervised.max(1)
            _, labels_max = labels.max(1)
            correct_predictions_d = predictions.eq(labels_max).sum().item()

            one_hot_predictions = F.one_hot(predictions, num_classes=6).float()
            one_hot_labels = labels

            test_data_count += one_hot_labels.size(0)
            test_corrects += correct_predictions_d
            test_predictions_f1.extend(one_hot_predictions.detach().cpu().max(1)[1])
            test_true_labels_f1.extend(one_hot_labels.detach().cpu().max(1)[1])

        test_accuracy = test_corrects / test_data_count
        test_true_labels_f1_np = np.array(test_true_labels_f1)
        test_predictions_f1_np = np.array(test_predictions_f1)
        test_f1 = f1_score(
            test_true_labels_f1_np, test_predictions_f1_np, average="micro"
        )
    print(f"Test Accuracy equals: {test_accuracy}")
    print(f"Test f1 score equals: {test_f1}")


def test_gan_bert(
    bert_model_name: str,
    transformer_path: str,
    generator_path: str,
    discriminator_path: str,
    bow_mode: bool,
    test_dataloader: DataLoader,
):
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer, _ = get_bert_model(bert_model_name)
    transformer.load_state_dict(torch.load(transformer_path))
    transformer.to(device)
    transformer.eval()
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(discriminator_path))
    discriminator.to(device)
    discriminator.eval()
    if bow_mode:
        generator, _ = get_bert_model(bert_model_name)
    else:
        generator = Generator1()
    generator.load_state_dict(torch.load(generator_path))
    generator.to(device)
    generator.eval()

    test_corrects = 0.0
    test_data_count = 0.0

    test_predictions_f1 = []
    test_true_labels_f1 = []

    # Validation loop
    generator.eval()
    discriminator.eval()
    transformer.eval()

    for batch in test_dataloader:
        # Unpack this training batch from our dataloader.
        encoded_input = batch[0].to(device)
        encoded_attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        is_supervised = batch[3].numpy().tolist()
        if bow_mode:
            encoded_bow = batch[4].to(device)
            encoded_bow_attention = batch[5].to(device)

        with torch.no_grad():
            supervised_indices = [item for item in is_supervised if item == 1]
            # supervised_indices   = torch.nonzero(is_supervised == 1).squeeze()
            unsupervised_indices = [item for item in is_supervised if item == 0]
            real_batch_size = encoded_input.shape[0]

            # Encode real data in the Transformer
            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            hidden_states = model_outputs[-1]

            # Define noise_size as the same size as the encoded_input
            noise_size = 100
            noise = torch.zeros((real_batch_size, noise_size)).uniform_(0, 1)

            # Train Generator1
            if bow_mode:
                generator_outputs = transformer(
                    encoded_bow, attention_mask=encoded_bow_attention
                )[-1]
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
            if len(supervised_indices) != 0:
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
            one_hot_predictions = F.one_hot(predictions, num_classes=7)
            one_hot_labels = F.one_hot(labels_index, num_classes=7).float()
            # -------------------------------------------------------------------------

            test_data_count += one_hot_labels.size(0)
            test_corrects += correct_predictions_d
            test_predictions_f1.extend(one_hot_predictions.detach().cpu().max(1)[1])
            test_true_labels_f1.extend(one_hot_labels.detach().cpu().max(1)[1])

    # Calculate average loss and accuracy
    test_accuracy = test_corrects / test_data_count
    test_true_labels_f1_np = np.array(test_true_labels_f1)
    test_predictions_f1_np = np.array(test_predictions_f1)
    test_f1 = f1_score(test_true_labels_f1_np, test_predictions_f1_np, average="micro")

    print(f"Test Accuracy equals: {test_accuracy}")
    print(f"Test f1 score equals: {test_f1}")
