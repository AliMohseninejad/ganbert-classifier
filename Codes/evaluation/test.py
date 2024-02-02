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
) -> (float, float):
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
    return test_accuracy, test_f1


def test_gan_bert(
    bert_model_name: str,
    transformer_path: str,
    generator_path: str,
    discriminator_path: str,
    bow_mode: bool,
    test_dataloader: DataLoader,
) -> (float, float):
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

    test_corrects = 0.0
    test_data_count = 0.0

    test_predictions_f1 = []
    test_true_labels_f1 = []

    # Validation loop
    discriminator.eval()
    transformer.eval()

    for batch in test_dataloader:
        # Unpack this training batch from our dataloader.
        encoded_input = batch[0].to(device)
        encoded_attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            real_batch_size = encoded_input.shape[0]

            # Encode real data in the Transformer
            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            hidden_states = model_outputs[-1]

            # Train Discriminator
            discriminator_input = hidden_states

            features, logits, probabilities = discriminator(discriminator_input)
            filtered_logits = logits[:, 0:-1]

            # Calculate the number of correct predictions for real and fake examples
            real_prediction_supervised = filtered_logits
            _, predictions = real_prediction_supervised.max(1)
            _, labels_max = labels.max(1)
            correct_predictions_d = predictions.eq(labels_max).sum().item()

            one_hot_predictions = F.one_hot(predictions, num_classes=6).float()
            one_hot_labels = labels

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

    return test_accuracy, test_f1
