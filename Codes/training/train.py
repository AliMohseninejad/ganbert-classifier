from model.discriminator import Discriminator
from model.generator1 import Generator1
from training.loss import GeneratorLossFunction, DiscriminatorLossFunction
from typing import *
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertModel
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np

# =============================================================================


def train_vanilla_classier(
    transformer: BertModel,
    classifier: Discriminator,
    optimizer: optim.Optimizer,
    epochs: int,
    scheduler: Union[LRScheduler, None],
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    transformer_path: str,
    classifier_path: str,
) -> Tuple[BertModel, Discriminator, List[Dict[str, float]]]:
    """Main function to train the simple BERT + Discriminator model. In this
    code, the model is trained and validated. In the end the best model is saved
    on disk.

    Args:
        transformer (BertModel) : The pre-trained BERT model
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
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    transformer.to(device)
    classifier.to(device)

    loss_function_train = nn.CrossEntropyLoss(ignore_index=-1)
    loss_function_validation = nn.CrossEntropyLoss(ignore_index=-1)

    results = []
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = 0.0
        validation_loss = 0.0
        # --------------------------
        train_accuracy = 0.0
        validation_accuracy = 0.0
        # --------------------------
        train_f1 = 0.0
        validation_f1 = 0.0
        # --------------------------
        corrects = 0.0
        validation_corrects = 0.0
        # --------------------------
        data_count = 0.0
        validation_data_count = 0.0

        predictions_f1 = []
        validation_predictions_f1 = []
        true_labels_f1 = []
        validation_true_labels_f1 = []

        classifier.train()
        transformer.train()

        for batch_i, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            # Unpack this training batch from our dataloader.
            encoded_input = batch[0].to(device)
            encoded_attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            is_supervised = batch[3].to(device)

            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            features, logits, probabilities = classifier(model_outputs[-1])

            # ------------------------------------------------
            real_prediction_supervised = probabilities
            _, predictions = real_prediction_supervised.max(1)
            _, labels_max = labels.max(1)
            correct_predictions_d = predictions.eq(labels_max).sum().item()
            one_hot_predictions = F.one_hot(predictions, num_classes=6).float()
            one_hot_labels = labels

            loss = loss_function_train(logits, one_hot_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            data_count += one_hot_labels.size(0)
            corrects += correct_predictions_d
            predictions_f1.extend(one_hot_predictions.detach().cpu().max(1)[1])
            true_labels_f1.extend(one_hot_labels.detach().cpu().max(1)[1])

            # print("============================================")
            # print("corrects:" , corrects , "data_count:", data_count, "in batch :" , batch_i, "out of", str(len(train_dataloader)))
            # print("============================================")

        train_loss /= len(train_dataloader)
        train_accuracy = corrects / data_count
        true_labels_f1_np = np.array(true_labels_f1)
        predictions_f1_np = np.array(predictions_f1)
        train_f1 = f1_score(true_labels_f1_np, predictions_f1_np, average="micro")

        # Validation
        classifier.eval()
        transformer.eval()
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            encoded_input_val = batch[0].to(device)
            encoded_attention_mask_val = batch[1].to(device)
            labels_val = batch[2].to(device)
            is_supervised_val = batch[3].to(device)

            with torch.no_grad():
                model_outputs_val = transformer(
                    encoded_input_val, attention_mask=encoded_attention_mask_val
                )
                features_val, logits_val, probabilities_val = classifier(
                    model_outputs_val[-1]
                )

                real_prediction_supervised_val = probabilities_val
                _, predictions_val = real_prediction_supervised_val.max(1)
                _, labels_max_val = labels_val.max(1)
                correct_predictions_d_val = (
                    predictions_val.eq(labels_max_val).sum().item()
                )

                one_hot_predictions_val = F.one_hot(
                    predictions_val, num_classes=6
                ).float()
                one_hot_labels_val = labels_val

                loss_val = loss_function_validation(logits_val, one_hot_labels_val)

                validation_loss += loss_val
                validation_data_count += one_hot_labels_val.size(0)
                validation_corrects += correct_predictions_d_val
                validation_predictions_f1.extend(
                    one_hot_predictions_val.detach().cpu().max(1)[1]
                )
                validation_true_labels_f1.extend(
                    one_hot_labels_val.detach().cpu().max(1)[1]
                )

        validation_loss /= len(validation_dataloader)
        validation_accuracy = validation_corrects / validation_data_count
        validation_true_labels_f1_np = np.array(validation_true_labels_f1)
        validation_predictions_f1_np = np.array(validation_predictions_f1)
        validation_f1 = f1_score(
            validation_true_labels_f1_np,
            validation_predictions_f1_np,
            average="micro",
        )

        # Update best model
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save(classifier.state_dict(), classifier_path)
            torch.save(transformer.state_dict(), transformer_path)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        print("\n----------------------------------------------")
        # Print progress
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f} "
            f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}, Validation F1: {validation_f1:.4f}"
        )
        print("-----------------------------------------------\n")

        # Update results
        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_f1": validation_f1,
        }
        results.append(result)

    print(f"Best model saved at epoch {best_epoch}")

    return transformer, classifier, results


# ==================================================================================================


def train_gan(
    transformer: BertModel,
    generator: Union[Generator1, BertModel],
    discriminator: Discriminator,
    bow_mode: bool,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    epochs: int,
    generator_scheduler: Union[LRScheduler, None],
    discriminator_scheduler: Union[LRScheduler, None],
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    generator_path: str,
    transformer_path: str,
    discriminator_path: str,
) -> Tuple[BertModel, Generator1, Discriminator, List[Dict[str, float]]]:
    """Main function to train the BERT-GAN model. In this code, the model is
    trained and validated. In the end the best model is saved on disk.

    Args:
        transformer (BertModel): The pre-trained BERT model
        generator (Generator1): The Generator model
        discriminator (Discriminator): The Discriminator model
        bow_mode (bool): If True, The generator is a BERT model. Otherwise it is
        a Generator model.
        generator_optimizer (optim.Optimizer): The Generator optimizer
        discriminator_optimizer (optim.Optimizer): The Discriminator optimizer
        epochs (int): number of training epochs
        generator_scheduler (Union[LRScheduler, None]): The Generator learning
        rate scheduler. No scheduler if None.
        discriminator_scheduler (Union[LRScheduler, None]): The Generator
        learning rate scheduler. No scheduler if None.
        train_dataloader (DataLoader): training dataloader
        validation_dataloader (DataLoader): validation dataloader
        generator_path (str): The path to the folder where generator should be saved
        transformer_path (str): The path to the folder where transformer should be saved
        discriminator_path (str): The path to the folder where discriminator should be saved

    Returns:
        Tuple[BertModel, Generator, Discriminator, List[Dict[str, float]]]: A
        tuple containing the trained BERT model, The trained Generator, the
        trained Discriminator and a list of results.  Each element of the list
        is a dictionary containing train loss per epoch, validation loss per
        epoch, train accuracy, validation accuracy, ...
    """
    random.seed(42)
    results = []
    best_loss = float("inf")
    best_epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # ---------------------------------------------------------------------
    generator_loss_function_train = GeneratorLossFunction()
    discriminator_loss_function_train = DiscriminatorLossFunction()

    generator_loss_function_validation = GeneratorLossFunction()
    discriminator_loss_function_validation = DiscriminatorLossFunction()
    # ---------------------------------------------------------------------

    generator.to(device)
    discriminator.to(device)
    transformer.to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        validation_loss = 0.0
        # ------------------------
        train_accuracy = 0.0
        validation_accuracy = 0.0
        # ------------------------
        train_f1 = 0.0
        validation_f1 = 0.0
        # ------------------------
        corrects = 0.0
        validation_corrects = 0.0
        data_count = 0.0
        validation_data_count = 0.0

        predictions_f1 = []
        validation_predictions_f1 = []
        true_labels_f1 = []
        validation_true_labels_f1 = []

        # Training loop
        generator.train()
        discriminator.train()
        transformer.train()

        # train
        for batch_i, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # Unpack this training batch from our dataloader.
            encoded_input = batch[0].to(device)
            encoded_attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            is_supervised = batch[3].numpy().tolist()
            if bow_mode:
                encoded_bow = batch[4].to(device)
                encoded_bow_attention = batch[5].to(device)

            supervised_indices = [item for item in is_supervised if item == 1]
            # supervised_indices   = torch.nonzero(is_supervised == 1).squeeze()
            unsupervised_indices = [item for item in is_supervised if item == 0]
            # unsupervised_indices = torch.nonzero(is_supervised == 0).squeeze()
            real_batch_size = encoded_input.shape[0]
            # Encode real data in the Transformer
            model_outputs = transformer(
                encoded_input, attention_mask=encoded_attention_mask
            )
            hidden_states = model_outputs[-1]

            # Define noise_size as the same size as the encoded_input
            noise_size = 100

            noise = torch.zeros((real_batch_size, noise_size), device=device).uniform_(
                0, 1
            )

            # -------------------------------

            # Train Generator
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
            # if supervised_indices.shape[0] != 0 :
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
            one_hot_predictions = F.one_hot(predictions, num_classes=7).float()
            one_hot_labels = F.one_hot(labels_index, num_classes=7).float()
            # -------------------------------------------------------------------------
            # Finally, we separate the discriminator's output for the real and fake data

            features_list = torch.split(features, real_batch_size)
            Discriminator_real_features = features_list[0]
            Discriminator_fake_features = features_list[1]

            probability_list = torch.split(probabilities, real_batch_size)
            Discriminator_real_probability = probability_list[0]
            Discriminator_fake_probability = probability_list[1]
            # --------------------------------------------------------------------------

            discriminator_loss = discriminator_loss_function_train(
                labels,
                supervised_indices,
                unsupervised_indices,
                Discriminator_real_probability,
                Discriminator_fake_probability,
            )
            generator_loss = generator_loss_function_train(
                generator_outputs,
                Discriminator_real_features,
                Discriminator_fake_features,
            )
            generator_loss.backward(retain_graph=True)
            discriminator_loss.backward()
            generator_optimizer.step()
            discriminator_optimizer.step()

            train_loss += generator_loss.item() + discriminator_loss.item()
            data_count += one_hot_labels.size(0)
            corrects += correct_predictions_d
            predictions_f1.extend(one_hot_predictions.detach().cpu().max(1)[1])
            true_labels_f1.extend(one_hot_labels.detach().cpu().max(1)[1])

        train_loss /= len(train_dataloader)
        train_accuracy = corrects / data_count
        true_labels_f1_np = np.array(true_labels_f1)
        predictions_f1_np = np.array(predictions_f1)
        train_f1 = f1_score(true_labels_f1_np, predictions_f1_np, average="micro")

        # Validation loop
        generator.eval()
        discriminator.eval()
        transformer.eval()

        with torch.no_grad():
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                encoded_input = batch[0].to(device)
                encoded_attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                is_supervised = batch[3].numpy().tolist()
                if bow_mode:
                    encoded_bow = batch[4].to(device)
                    encoded_bow_attention = batch[5].to(device)

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

                noise = torch.zeros(
                    real_batch_size, noise_size, device=device
                ).uniform_(0, 1)

                # -------------------------------

                # Train Generator
                if bow_mode:
                    generator_outputs = transformer(
                        encoded_bow, attention_mask=encoded_bow_attention
                    )[-1]
                else:
                    generator_outputs = generator(noise)

                # ------------------------------------

                # Train Discriminator
                discriminator_input = torch.cat(
                    [generator_outputs, hidden_states], dim=0
                )

                features, logits, probabilities = discriminator(discriminator_input)

                # Calculate the number of correct predictions for real and fake examples
                fake_predictions = probabilities[:real_batch_size]
                real_predictions = probabilities[real_batch_size:]

                # ------------------------------------------------
                # if supervised_indices.shape[0] != 0 :
                if len(supervised_indices) != 0:
                    real_prediction_supervised = real_predictions[supervised_indices]
                    sup_fake_probabilities = torch.cat(
                        [fake_predictions, real_prediction_supervised], dim=0
                    )
                    _, predictions = sup_fake_probabilities.max(1)
                    fake_labels = torch.zeros_like(fake_predictions)
                    fake_labels[:, -1] = 1
                    fake_labels.to(fake_predictions.device)
                    all_labels = torch.cat(
                        [fake_labels, labels[supervised_indices]], dim=0
                    )
                else:
                    _, predictions = fake_predictions.max(1)
                    all_labels = torch.zeros_like(fake_predictions)
                    all_labels[:, -1] = 1
                    all_labels.to(fake_predictions.device)

                _, labels_index = all_labels.max(1)
                correct_predictions_d = predictions.eq(labels_index).sum().item()
                one_hot_predictions = F.one_hot(predictions, num_classes=7).float()
                one_hot_labels = F.one_hot(labels_index, num_classes=7).float()
                # -------------------------------------------------------------------------
                # Finally, we separate the discriminator's output for the real and fake data

                features_list = torch.split(features, real_batch_size)
                Discriminator_real_features = features_list[0]
                Discriminator_fake_features = features_list[1]

                probability_list = torch.split(probabilities, real_batch_size)
                Discriminator_real_probability = probability_list[0]
                Discriminator_fake_probability = probability_list[1]
                # --------------------------------------------------------------------------

                discriminator_loss = discriminator_loss_function_validation(
                    labels,
                    supervised_indices,
                    unsupervised_indices,
                    Discriminator_real_probability,
                    Discriminator_fake_probability,
                )
                generator_loss = generator_loss_function_validation(
                    generator_outputs,
                    Discriminator_real_features,
                    Discriminator_fake_features,
                )

                validation_loss += generator_loss.item() + discriminator_loss.item()
                validation_data_count += one_hot_labels.size(0)
                validation_corrects += correct_predictions_d
                validation_predictions_f1.extend(
                    one_hot_predictions.detach().cpu().max(1)[1]
                )
                validation_true_labels_f1.extend(
                    one_hot_labels.detach().cpu().max(1)[1]
                )

        # Calculate average loss and accuracy
        validation_loss /= len(train_dataloader)
        validation_accuracy = validation_corrects / validation_data_count
        validation_true_labels_f1_np = np.array(validation_true_labels_f1)
        validation_predictions_f1_np = np.array(validation_predictions_f1)
        validation_f1 = f1_score(
            validation_true_labels_f1_np, validation_predictions_f1_np, average="micro"
        )

        # Update best model
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(transformer.state_dict(), transformer_path)

        # ------------------------------------------------------------------------------------------
        # Update results
        result = {
            "epoch": epoch,
            # -------------------------------------------
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            # -------------------------------------------
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_f1": validation_f1,
        }
        results.append(result)
        # ------------------------------------------------------------------------------------------

        # Update learning rate scheduler
        if generator_scheduler is not None:
            generator_scheduler.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()

        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f},Train_f1: {train_f1: .4f}"
            f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}, Validation_f1: {validation_f1:.4f}"
        )

    print(f"Best model saved at epoch {best_epoch}")

    return transformer, generator, discriminator, results
