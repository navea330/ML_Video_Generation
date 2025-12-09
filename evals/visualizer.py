# visualizer.py
import matplotlib.pyplot as plt
import os

def plot_training_curves(train_losses, val_losses, val_accuracies, save_path="training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:cyan', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, label='Val Acc', color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
