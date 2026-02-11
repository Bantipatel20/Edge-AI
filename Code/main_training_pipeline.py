# !pip install seaborn
import os
import random
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

# ----------------------------------------------------------
# GLOBAL CONFIG
# ----------------------------------------------------------
random.seed(42)
torch.manual_seed(42)

CLASS_NAMES = ["Mixed", "Not_Mixed"]
PLOTS_DIR = "Code/plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------------------------------------
# STEP 1: DATASET SPLIT
# ----------------------------------------------------------
def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    classes = ["Mixed", "Not_Mixed"]

    for cls in classes:
        img_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "Training": images[:n_train],
            "Validation": images[n_train:n_train + n_val],
            "Test": images[n_train + n_val:]
        }

        for split, files in splits.items():
            out_dir = os.path.join(output_dir, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(img_dir, f), out_dir)

# ----------------------------------------------------------
# STEP 2: CUSTOM DATASET (9 CHANNEL)
# ----------------------------------------------------------
class DrugDataset9Ch(Dataset):
    def __init__(self, root_dir, label_map):
        self.paths, self.labels = [], []
        self.transform = transforms.Resize((512, 512))

        for cls, label in label_map.items():
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(cls_path, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(self.transform(img))

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        combined = np.concatenate([img, hsv, lab], axis=2)
        combined = combined.transpose(2, 0, 1) / 255.0

        return torch.tensor(combined, dtype=torch.float32), torch.tensor(self.labels[idx])

# ----------------------------------------------------------
# STEP 3: MODIFY DENSENET INPUT CHANNELS
# ----------------------------------------------------------
def modify_densenet_input_channels(model, in_channels=9):
    conv = model.features.conv0
    new_conv = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False
    )

    with torch.no_grad():
        new_conv.weight[:, :3] = conv.weight
        for i in range(3, in_channels):
            new_conv.weight[:, i:i+1] = conv.weight[:, i % 3:i % 3 + 1]

    model.features.conv0 = new_conv
    return model

def train(model, train_loader, val_loader, device, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # >>> NEW: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            # >>> NEW: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        # >>> NEW: step scheduler each epoch
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_acc, val_loss, _, _, _ = evaluate(
            model, val_loader, device, criterion, verbose=False
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_densenet_9ch.pth")

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return train_losses, train_accs, val_losses, val_accs



def evaluate(model, loader, device, criterion=None, verbose=True):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            if criterion is not None:
                total_loss += criterion(out, y).item()

            probs = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    loss = total_loss / len(loader) if criterion else None

    if verbose:
        print("Accuracy:", acc)
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1:", f1_score(y_true, y_pred))

    return acc, loss, y_true, y_pred, y_prob


# ----------------------------------------------------------
# STEP 6: ALL PLOTS (300 DPI)
# ----------------------------------------------------------
def save_plot(name):
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{name}.png", dpi=300)
    plt.close()

def plot_class_distribution(dataset):
    labels = [y for _, y in dataset]
    counts = np.bincount(labels)
    plt.bar(CLASS_NAMES, counts)
    plt.title("Class Distribution")
    save_plot("class_distribution")

def plot_training_curves(losses, accs):
    plt.plot(losses); plt.title("Training Loss"); save_plot("training_loss")
    plt.plot(accs); plt.title("Training Accuracy"); save_plot("training_accuracy")

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for mat, name in [(cm, "confusion_matrix"),
                      (cmn, "confusion_matrix_normalized")]:
        sns.heatmap(mat, annot=True,
                    fmt=".2f" if "normalized" in name else "d",
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    cmap="Blues")
        plt.title(name.replace("_", " ").title())
        save_plot(name)

def plot_classwise_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]

    for i, m in enumerate(metrics):
        plt.bar(np.arange(2) + i*0.25,
                [report[c][m] for c in CLASS_NAMES],
                width=0.25, label=m)

    plt.xticks(np.arange(2) + 0.25, CLASS_NAMES)
    plt.legend()
    plt.title("Class-wise Metrics")
    save_plot("classwise_metrics")

def plot_roc_pr(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
    plt.legend(); plt.title("ROC Curve"); save_plot("roc_curve")

    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(r, p); plt.title("Precision-Recall Curve")
    save_plot("precision_recall_curve")

def plot_misclassified(model, loader, device):
    model.eval()
    imgs = []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for i in range(len(y)):
                if preds[i] != y[i]:
                    imgs.append(x[i][:3])
                if len(imgs) == 16:
                    break
    grid = make_grid(imgs, nrow=4, normalize=True)
    plt.imshow(grid.permute(1,2,0)); plt.axis("off")
    plt.title("Misclassified Samples")
    save_plot("misclassified_samples")

def plot_train_val_accuracy(train_accs, val_accs):
    plt.figure()
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    save_plot("train_val_accuracy")


def plot_train_val_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    save_plot("train_val_loss")


def plot_eval_metrics_per_epoch(model, val_loader, device):
    acc, loss, y_true, y_pred, y_prob = evaluate(
        model, val_loader, device, verbose=False
    )

    plt.figure()
    plt.bar(
        ["Accuracy", "Precision", "Recall", "F1"],
        [
            acc,
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
        ],
    )
    plt.ylim(0, 1.05)
    plt.title("Final Validation Metrics")
    save_plot("final_validation_metrics")


class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.clone()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone()

    def generate(self, model, x, class_idx):
        model.zero_grad()
        out = model(x)

        score = out[:, class_idx]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def plot_gradcam_attention(model, loader, device):
    model.eval()
    cam = GradCAM(model, model.features[-1])

    x, y = next(iter(loader))
    x = x.to(device)

    with torch.no_grad():
        preds = model(x).argmax(1)

    cam_map = cam.generate(model, x, preds[0].item())[0].cpu().numpy()
    img = x[0][:3].permute(1,2,0).cpu().numpy()

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_map),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = 0.6 * img + 0.4 * heatmap

    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM Attention Overlay")
    save_plot("gradcam_attention")


# +
# ----------------------------------------------------------
# MAIN (UPDATED)
# ----------------------------------------------------------
if __name__ == "__main__":

    # ------------------------
    # Paths
    # ------------------------
    dataset_dir = "../Dataset/"
    output_dir = "../output/"

    # ------------------------
    # Dataset split
    # ------------------------
    split_dataset(dataset_dir, output_dir)

    label_map = {
        "Mixed": 0,
        "Not_Mixed": 1
    }

   
    train_ds = DrugDataset9Ch(
        os.path.join(output_dir, "Training"),
        label_map
    )

    val_ds = DrugDataset9Ch(
        os.path.join(output_dir, "Validation"),
        label_map
    )

    test_ds = DrugDataset9Ch(
        os.path.join(output_dir, "Test"),
        label_map
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")


    model = models.densenet121(pretrained=True)
    model = modify_densenet_input_channels(model, in_channels=9)
    model.classifier = nn.Linear(
        model.classifier.in_features, 2
    )
    model.to(device)

    # ------------------------
    # Train
    # ------------------------
    train_losses, train_accs, val_losses, val_accs = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=15
    )


    model.load_state_dict(
        torch.load("best_densenet_9ch.pth", map_location=device)
    )


    test_acc, _, y_true, y_pred, y_prob = evaluate(
        model,
        test_loader,
        device,
        verbose=True
    )

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

  
    plot_class_distribution(train_ds)
    plot_training_curves(train_losses, train_accs)

    plot_train_val_accuracy(train_accs, val_accs)
    plot_train_val_loss(train_losses, val_losses)

    plot_confusion(y_true, y_pred)
    plot_classwise_metrics(y_true, y_pred)
    plot_roc_pr(y_true, y_prob)
    plot_misclassified(model, test_loader, device)
    plot_eval_metrics_per_epoch(model, val_loader, device)
    plot_gradcam_attention(model, test_loader, device)


