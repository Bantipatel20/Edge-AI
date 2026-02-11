import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_model_comparison(metrics_table, out_path):
    df = pd.DataFrame(metrics_table)
    ax = df.plot(kind='bar', figsize=(10,6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved model comparison chart to {out_path}")

if __name__ == "__main__":
    # Example metrics table
    metrics_table = {
        'DenseNet': [0.98, 0.97, 0.96, 0.97],
        'VGG16': [0.95, 0.94, 0.93, 0.94],
        'Inception': [0.96, 0.95, 0.94, 0.95]
    }
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    df = pd.DataFrame(metrics_table, index=metric_names)
    plot_model_comparison(df, './plots/model_comparison.png')
