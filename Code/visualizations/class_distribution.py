import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(class_counts, save_path):
    plt.figure(figsize=(5,4), dpi=300)
    sns.barplot(
        x=list(class_counts.keys()),
        y=list(class_counts.values())
    )
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
