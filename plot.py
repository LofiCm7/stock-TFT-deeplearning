import os
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def plot_training_curves(history, save_dir=FIGURES_DIR):
    """Plot train/val loss, IC, and ICIR curves."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['ic'], marker='o', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IC')
    axes[1].set_title('IC')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    axes[2].plot(epochs, history['icir'], marker='o', markersize=3)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('ICIR')
    axes[2].set_title('ICIR')
    axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved to {path}")
