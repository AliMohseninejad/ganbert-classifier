from   model.discriminator      import Discriminator
from   model.generator1         import Generator
from   loss                     import GeneratorLossFunction, DiscriminatorLossFunction
from   typing                   import *
from   torch.optim.lr_scheduler import LRScheduler
from   transformers             import BertModel
from   torch.utils.data         import DataLoader
from   sklearn.metrics          import precision_score, recall_score, f1_score
import torch.nn.functional      as F
import torch.optim              as optim
import torch.nn                 as nn
import torch



def train_vanilla_classier(
    transformer          : BertModel               ,
    classifier           : Discriminator           ,
    optimizer            : optim.Optimizer         ,
    epochs               : int                     ,
    scheduler            : Union[LRScheduler, None],
    train_dataloader     : DataLoader              ,
    validation_dataloader: DataLoader              ,
    transformer_path     : str                     ,
    classifier_path      : str                     ,
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)
    classifier.to(device)

    optimizer = optim.Optimizer([{'params': transformer.parameters()}, {'params': classifier.parameters()}])

    loss_function = nn.CrossEntropyLoss()
    
    results    = []
    best_loss  = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        
        train_loss          = 0.0
        train_accuracy      = 0.0
        train_f1            = 0.0
        #------------------------
        validation_loss     = 0.0
        validation_accuracy = 0.0
        validation_f1       = 0.0
        
        classifier.train()
        transformer.train() 
        
        for batch in train_dataloader:
            
            optimizer.zero_grad()
            # Unpack this training batch from our dataloader. 
            encoded_input          = batch[0].to(device)
            encoded_attention_mask = batch[1].to(device)
            labels                 = batch[2].to(device)
            #if bow_mode:
            #    encoded_bow = batch[3].to(device)
                
            outputs = classifier(transformer(encoded_input, attention_mask=encoded_attention_mask))
            loss    = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss     += loss.item()
            train_accuracy += (outputs.argmax(dim=1) == labels).sum().item()
            train_f1 += f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='binary')

            
        # Validation
        classifier.eval()
        transformer.eval()
        
        with torch.no_grad():
            for batch in validation_dataloader:
                
                encoded_input          = batch[0].to(device)
                encoded_attention_mask = batch[1].to(device)
                labels                 = batch[2].to(device)
                #if bow_mode:
                #    encoded_bow = batch[3].to(device)
                    
                outputs = classifier(transformer(encoded_input, attention_mask=encoded_attention_mask))
                loss    = loss_function(outputs, labels)
                validation_loss     += loss.item()
                validation_accuracy += (outputs.argmax(dim=1) == labels).sum().item()
                validation_f1       += f1_score(labels.cpu().numpy(), outputs.argmax(dim=1).cpu().numpy(), average='binary')


        # Calculate average loss and accuracy
        train_loss          /= len(train_dataloader.dataset)
        train_accuracy      /= len(train_dataloader.dataset)
        train_f1            /= len(train_dataloader.dataset)
        validation_loss     /= len(validation_dataloader.dataset)
        validation_accuracy /= len(validation_dataloader.dataset)
        validation_f1       /= len(validation_dataloader.dataset)
        
        # Update best model
        if validation_loss < best_loss:
            best_loss  = validation_loss
            best_epoch = epoch
            torch.save(classifier.state_dict() , classifier_path)
            torch.save(transformer.state_dict(), transformer_path)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Print progress
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}"
            f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}, Validation F1: {validation_f1:.4f}"
        )

        # Update results
        result = {
            "epoch"               : epoch               ,
            "train_loss"          : train_loss          ,
            "train_accuracy"      : train_accuracy      ,
            "train_f1"            : train_f1            ,
            "validation_loss"     : validation_loss     ,
            "validation_accuracy" : validation_accuracy ,
            "validation_f1"       : validation_f1       ,
        }
        results.append(result)

    print(f"Best model saved at epoch {best_epoch}")
    
    return transformer, classifier, results

    

#==================================================================================================

def train_gan(
    transformer                : BertModel                  ,
    generator                  : Union[Generator, BertModel],
    discriminator              : Discriminator              ,
    bow_mode                   : bool                       ,
    generator_optimizer        : optim.Optimizer            ,
    discriminator_optimizer    : optim.Optimizer            ,
    generator_loss_function    : GeneratorLossFunction      ,
    discriminator_loss_function: DiscriminatorLossFunction  ,
    epochs                     : int                        ,
    generator_scheduler        : Union[LRScheduler, None]   ,
    discriminator_scheduler    : Union[LRScheduler, None]   ,
    train_dataloader           : DataLoader                 ,
    validation_dataloader      : DataLoader                 ,
    generator_path             : str                        ,
    transformer_path           : str                        ,
    discriminator_path         : str                        ,
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
    
    results    = []
    best_loss  = float('inf')
    best_epoch = 0
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator_loss_function     = GeneratorLossFunction()
    discriminator_loss_function = DiscriminatorLossFunction()
    
    generator.to(device)
    discriminator.to(device)
    transformer.to(device)
    
    for epoch in range(epochs):
        
        train_loss          = 0.0
        validation_loss     = 0.0
        #------------------------
        train_accuracy      = 0.0
        validation_accuracy = 0.0
        #------------------------
        train_f1            = 0.0
        validation_f1       = 0.0
        #------------------------
        total_examples      = 0.0

        # Training loop
        generator.train()
        discriminator.train()
        transformer.train()
        
        #train
        for batch in train_dataloader:
            
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
                        
            # Unpack this training batch from our dataloader. 
            encoded_input          = batch[0].to(device)
            encoded_attention_mask = batch[1].to(device)
            labels                 = batch[2].to(device)
            if bow_mode:
                encoded_bow = batch[3].to(device)

            real_batch_size = encoded_input.shape[0]
            
            # Encode real data in the Transformer
            model_outputs = transformer(encoded_input, attention_mask=encoded_attention_mask)
            hidden_states = model_outputs[-1]

            # Define noise_size as the same size as the encoded_input
            noise_size = encoded_input.shape[1]

            noise = torch.zeros(real_batch_size, noise_size, device=device).uniform_(0, 1)
        
            #-------------------------------
            
            # Train Generator
            if bow_mode:
                generator_outputs = transformer(encoded_bow)  
            else:
                generator_outputs = generator(noise)
                
            generator_loss = generator_loss_function(generator_outputs, real_features, generated_features)
            
            generator_loss.backward()
            generator_optimizer.step()

            #------------------------------------
            
            # Train Discriminator
            discriminator_input = torch.cat([hidden_states, generator_outputs], dim=0)
            
            features, logits, probabilities = discriminator(discriminator_input)
            
            # Calculate the number of correct predictions for real and fake examples
            real_predictions = (probabilities[:real_batch_size] >= 0.5).long()
            fake_predictions = (probabilities[real_batch_size:] < 0.5).long()

            # Calculate the number of correct predictions for the entire batch
            correct_predictions = (real_predictions == labels).sum() + (fake_predictions == 1 - labels).sum()

            # Increment the total number of examples
            total_examples += real_batch_size
            
            
            #-------------------------------------------------------------------------
            # Finally, we separate the discriminator's output for the real and fake data
            
            features_list    = torch.split(features, real_batch_size)
            Discriminator_real_features    = features_list[0]
            Discriminator_fake_features    = features_list[1]

            logits_list      = torch.split(logits, real_batch_size)
            Discriminator_real_logits      = logits_list[0]
            Discriminator_fake_logits      = logits_list[1]
            
            probability_list = torch.split(probabilities, real_batch_size)
            Discriminator_real_probability = probability_list[0]
            Discriminator_fake_probability = probability_list[1]
            #--------------------------------------------------------------------------
            
            # Discriminator's LOSS estimation
            logits               = Discriminator_real_logits[:,0:-1]
            logits_probabilities = F.log_softmax(logits, dim=-1)
            
            #-------------------------------------------------------------------------
            
            discriminator_loss = discriminator_loss_function(labels, Discriminator_real_probability, Discriminator_fake_probability) 

            discriminator_loss.backward()
            discriminator_optimizer.step()

            train_loss     += (generator_loss.item() + discriminator_loss.item())
            train_accuracy += correct_predictions.item()
            train_f1       += f1_score(labels.cpu().numpy(), real_predictions.cpu().numpy(), average='binary')

        train_loss     /= len(train_dataloader.dataset)
        train_accuracy /= len(train_dataloader.dataset)
        train_f1       /= len(train_dataloader.dataset)
        
        # Validation loop
        generator.eval()
        discriminator.eval()
        transformer.eval()
        
        with torch.no_grad():
            for batch in validation_dataloader:
                
                encoded_input          = batch[0].to(device)
                encoded_attention_mask = batch[1].to(device)
                labels                 = batch[2].to(device)
                if bow_mode:
                    encoded_bow = batch[3].to(device)

                real_batch_size = encoded_input.shape[0]

                if bow_mode:
                    generator_outputs, real_features, generated_features = transformer(encoded_bow)  
                else:
                    generator_outputs, real_features, generated_features = generator(noise)
                    
                generator_loss_function(generator_outputs, real_features, generated_features)
                
                real_predictions = (generator_outputs[:real_batch_size] >= 0.5).long()
                fake_predictions = (generator_outputs[real_batch_size:] < 0.5).long()
                correct_predictions_g = (real_predictions == labels).sum() + (fake_predictions == 1 - labels).sum()

                # Discriminator Validation
                discriminator_input = torch.cat([hidden_states, generator_outputs], dim=0)

                features, logits, probabilities = discriminator(discriminator_input)

                real_predictions = (probabilities[:real_batch_size] >= 0.5).long()
                fake_predictions = (probabilities[real_batch_size:] < 0.5).long()
                correct_predictions_d = (real_predictions == labels).sum() + (fake_predictions == 1 - labels).sum()

                discriminator_loss = discriminator_loss_function(labels, probabilities)

                validation_loss     += (discriminator_loss.item() + generator_loss.item()          )
                validation_accuracy += (correct_predictions_d.item() + correct_predictions_g.item())
                validation_f1       += f1_score(labels.cpu().numpy(), real_predictions.cpu().numpy(), average='binary') 

        # Calculate average loss and accuracy
        validation_loss     /= len(validation_dataloader.dataset)
        validation_accuracy /= len(validation_dataloader.dataset)
        validation_f1       /= len(validation_dataloader.dataset)

        # Update best model
        if validation_loss < best_loss:
            best_loss  = validation_loss
            best_epoch = epoch
            torch.save(generator.state_dict()     , generator_path     )
            torch.save(discriminator.state_dict() , discriminator_path )
            torch.save(transformer.state_dict()   , transformer_path   )
        
        #------------------------------------------------------------------------------------------
        # Update results
        result = {
            "epoch"               : epoch,
            #-------------------------------------------
            "train_loss"          : train_loss,
            "train_accuracy"      : train_accuracy,
            "train_f1"            : train_f1,
            #-------------------------------------------
            "validation_loss"     : validation_loss,
            "validation_accuracy" : validation_accuracy,
            "validation_f1"       : validation_f1
        }
        results.append(result)
        #------------------------------------------------------------------------------------------
        
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
