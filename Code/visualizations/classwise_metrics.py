import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def plot_classwise_metrics(y_true, y_pred, labels, save_path):
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True
    )

    precision = [report[l]["precision"] for l in labels]
    recall = [report[l]["recall"] for l in labels]
    f1 = [report[l]["f1-score"] for l in labels]

    x = np.arange(len(labels))
    plt.figure(figsize=(6,4), dpi=300)
    plt.bar(x-0.25, precision, 0.25, label="Precision")
    plt.bar(x, recall, 0.25, label="Recall")
    plt.bar(x+0.25, f1, 0.25, label="F1-score")

    plt.xticks(x, labels)
    plt.ylim(0,1.05)
    plt.title("Class-wise Metrics")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
