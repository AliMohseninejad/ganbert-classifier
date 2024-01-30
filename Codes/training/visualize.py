import matplotlib.pyplot as plt

def plot_results(results, title_suffix):
    epochs              = [result["epoch"]                                   for result in results]
    train_loss          = [result["train_loss"].detach().cpu().numpy()       for result in results]
    validation_loss     = [result["validation_loss"].detach().cpu().numpy()  for result in results]
    train_accuracy      = [result["train_accuracy"]                          for result in results]
    validation_accuracy = [result["validation_accuracy"]                     for result in results]
    train_f1            = [result["train_f1"]                                for result in results]
    validation_f1       = [result["validation_f1"]                           for result in results]
    #----------------------------------------------------------------------------------------------
    # Plot Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss     , label='Train Loss'     , marker='o', linestyle='--', color= "#f70d1a", linewidth=2) #vivid red
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='o', linestyle='--', color= "#002147", linewidth=2) #oxford blue
    plt.title('Training and Validation Loss' + title_suffix)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.4, color='#b0b0b0')  # Add thin dashed grid lines
    plt.tight_layout()
    #----------------------------------------------------------------------------------------------
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracy     , label='Train Accuracy'     , marker='o', linestyle='-', color="#f70d1a", linewidth=2) #vivid red
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o', linestyle='-', color="#002147", linewidth=2) #oxford blue
    plt.title('Training and Validation Accuracy' + title_suffix)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.4, color='#b0b0b0')  # Add thin dashed grid lines
    plt.tight_layout()
    #----------------------------------------------------------------------------------------------
    # plot F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1     , label='Train F1 Score'     , marker='o', linestyle='-', color="#f70d1a", linewidth=2) #vivid red
    plt.plot(epochs, validation_f1, label='Validation F1 Score', marker='o', linestyle='-', color="#002147", linewidth=2) #oxford blue
    plt.title('Training and Validation F1 score' + title_suffix)
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.4, color='#b0b0b0')  # Add thin dashed grid lines
    plt.tight_layout()


    plt.show()

