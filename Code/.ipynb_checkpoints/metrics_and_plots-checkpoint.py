import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

def save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_roc_curve(y_true, y_scores, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_pr_curve(y_true, y_scores, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_epoch_metrics_plot(metrics_dict, out_path):
    plt.figure(figsize=(10,6))
    for key, values in metrics_dict.items():
        plt.plot(range(1, len(values)+1), values, label=key)
    plt.title('Evaluation Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_metrics_bar_chart(metrics, labels, out_path):
    plt.figure()
    plt.bar(labels, metrics, color='skyblue')
    plt.ylim(0.0, 1.05)
    plt.title('Model Evaluation Metrics')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Example usage:
if __name__ == "__main__":
    # Dummy data for demonstration
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
    y_scores = [0.2, 0.8, 0.1, 0.4, 0.9, 0.3, 0.7, 0.6]
    labels = ['Mixed', 'Not Mixed']
    save_confusion_matrix(y_true, y_pred, labels, './plots/confusion_matrix.png')
    save_roc_curve(y_true, y_scores, './plots/roc_curve.png')
    save_pr_curve(y_true, y_scores, './plots/pr_curve.png')
    metrics_dict = {
        'Accuracy': [0.8, 0.85, 0.9],
        'Precision': [0.75, 0.8, 0.88],
        'Recall': [0.7, 0.82, 0.9],
        'F1-Score': [0.72, 0.81, 0.89]
    }
    save_epoch_metrics_plot(metrics_dict, './plots/metrics_per_epoch.png')
    save_metrics_bar_chart([0.9, 0.88, 0.85, 0.89], ['Accuracy', 'Precision', 'Recall', 'F1-Score'], './plots/metrics_bar_chart.png')
