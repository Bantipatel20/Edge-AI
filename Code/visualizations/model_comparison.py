import matplotlib.pyplot as plt
import pandas as pd

def plot_model_comparison(model_scores, metric_names, save_path):
    df = pd.DataFrame(model_scores, index=metric_names)
    plt.figure(figsize=(7,4), dpi=300)
    df.plot(kind="bar")
    plt.ylabel("Score")
    plt.ylim(0,1.05)
    plt.title("Model Comparison")
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
