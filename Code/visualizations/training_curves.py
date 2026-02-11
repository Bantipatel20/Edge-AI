import matplotlib.pyplot as plt

def plot_loss_curve(train_loss, val_loss, save_path):
    epochs = range(1, len(train_loss)+1)
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curve(train_acc, val_acc, save_path):
    epochs = range(1, len(train_acc)+1)
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
