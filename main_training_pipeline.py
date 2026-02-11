import os
import random
import shutil
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# --- Step 1: Split dataset into train, val, test ---
# Assumes dataset_dir/Mixed and dataset_dir/Not Mixed contain images

def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = ['Mixed', 'Not_Mixed']
    for cls in classes:
        img_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]  
        test_imgs = images[n_train+n_val:]
        for split, split_imgs in zip(['Training', 'Validation', 'Test'], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                shutil.copy(os.path.join(img_dir, img), os.path.join(split_dir, img))

# --- Step 2: Custom Dataset Class ---
class DrugDataset9Ch(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform or transforms.Compose([transforms.Resize((512, 512))])
        for cls in label_map:
            class_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label_map[cls])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        rgb = np.array(img)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        combined = np.concatenate([rgb, hsv, lab], axis=2)
        combined = combined.transpose(2, 0, 1) / 255.0
        label = self.labels[idx]
        return torch.tensor(combined, dtype=torch.float32), torch.tensor(label)

# --- Step 3: Model Setup ---
def modify_densenet_input_channels(model, in_channels):
    first_conv = model.features.conv0
    new_conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=first_conv.out_channels,
                         kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride,
                         padding=first_conv.padding,
                         bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = first_conv.weight
        for i in range(3, in_channels):
            new_conv.weight[:, i:i+1] = first_conv.weight[:, i % 3:i % 3 + 1]
    model.features.conv0 = new_conv
    return model

# --- Step 4: Training & Evaluation ---
def train(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_densenet_9ch.pth')
            print("Best model saved!")

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    return acc

# +
# --- Step 5: Main Pipeline ---

if __name__ == "__main__":
    random.seed(42)
    dataset_dir = "../Dataset/Mixed/"  # <-- CHANGE THIS TO YOUR DATASET FOLDER
    output_dir = "../output"   # <-- CHANGE THIS TO WHERE YOU WANT SPLITS
    split_dataset(dataset_dir, output_dir)
    label_map = {'Mixed': 0, 'Not Mixed': 1}
    train_ds = DrugDataset9Ch(os.path.join(output_dir, 'Training'), label_map)
    val_ds = DrugDataset9Ch(os.path.join(output_dir, 'Validation'), label_map)
    test_ds = DrugDataset9Ch(os.path.join(output_dir, 'Test'), label_map)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet121(pretrained=True)
    model = modify_densenet_input_channels(model, in_channels=9)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.to(device)
    train(model, train_loader, val_loader, device, epochs=10)
    print("Final test set evaluation:")
    evaluate(model, test_loader, device)
