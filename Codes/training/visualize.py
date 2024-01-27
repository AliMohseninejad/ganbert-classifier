import matplotlib.pyplot as plt

def plot_results(results):
    epochs         = [result["epoch"]          for result in results]
    train_loss     = [result["train_loss"]     for result in results]
    val_loss       = [result["val_loss"]       for result in results]
    train_accuracy = [result["train_accuracy"] for result in results]
    val_accuracy   = [result["val_accuracy"]   for result in results]
    train_f1_score = [result["train_f1_score"] for result in results]
    val_f1_score   = [result["f1_score"]       for result in results]
    #----------------------------------------------------------------------------------------------
    # Plot Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss'     , marker='o', linestyle='--', color= "#f70d1a", linewidth=2) #vivid red
    plt.plot(epochs, val_loss,   label='Validation Loss', marker='o', linestyle='--', color= "#002147", linewidth=2) #oxford blue
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #----------------------------------------------------------------------------------------------
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracy, label='Train Accuracy'     , marker='o', linestyle='-', color="#f70d1a", linewidth=2) #vivid red
    plt.plot(epochs, val_accuracy,   label='Validation Accuracy', marker='o', linestyle='-', color="#002147", linewidth=2) #oxford blue
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #----------------------------------------------------------------------------------------------
    # plot F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1_score, label='Train F1 Score'   , marker='o', linestyle='-', color="#f70d1a", linewidth=2) #vivid red
    plt.plot(epochs, val_f1_score, label='Validation F1 Score', marker='o', linestyle='-', color="#002147", linewidth=2) #oxford blue
    plt.title('Training and Validation F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    #----------------------------------------------------------------------------------------------
    plt.grid(True, linestyle='--', linewidth=0.4, color='#b0b0b0')  # Add thin dashed grid lines
    plt.tight_layout()
    plt.show()

