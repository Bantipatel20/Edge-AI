import matplotlib.pyplot as plt
from .utils import ensure_dir

def save_misclassified(images, y_true, y_pred, labels, out_dir, max_samples=6):
    ensure_dir(out_dir)
    count = 0
    for i in range(len(images)):
        if y_true[i] != y_pred[i]:
            plt.figure(figsize=(3,3), dpi=300)
            plt.imshow(images[i])
            plt.title(f"GT:{labels[y_true[i]]} | Pred:{labels[y_pred[i]]}")
            plt.axis("off")
            plt.savefig(f"{out_dir}/error_{count}.png",
                        dpi=300, bbox_inches="tight")
            plt.close()
            count += 1
            if count >= max_samples:
                break
