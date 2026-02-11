import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

def plot_class_distribution(dataset_dir, out_path):
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    counts = {cls: len(os.listdir(os.path.join(dataset_dir, cls))) for cls in classes}
    plt.figure()
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette='viridis')
    plt.title('Class Distribution')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved class distribution plot to {out_path}")

def error_analysis(y_true, y_pred, out_path):
    errors = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    plt.figure()
    plt.hist(errors, bins=20, color='red', alpha=0.7)
    plt.title('Error Analysis: Misclassified Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved error analysis plot to {out_path}")

if __name__ == "__main__":
    dataset_dir = '../YourDatasetSplits/Training'  # <-- Update as needed
    plot_class_distribution(dataset_dir, './plots/class_distribution.png')
    # Example error analysis
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
    error_analysis(y_true, y_pred, './plots/error_analysis.png')
