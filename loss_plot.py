import matplotlib.pyplot as plt

def loss_plot(num_epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    plt.close()
    print("Training Loss plot saved as 'training_loss_plot.png'")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), val_losses, label='Evaluate Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evaluate Loss over Epochs')
    plt.legend()
    plt.savefig('evaluate_loss_plot.png')
    plt.close()
    print("Evaluate Loss plot saved as 'evaluate_loss_plot.png'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('train_valid_loss.png')
    plt.close()
    print("Loss plot saved as 'train_valid_loss.png'")