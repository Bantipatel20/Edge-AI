import matplotlib.pyplot as plt

def plot_gradcam(original, cam, overlay, save_path):
    plt.figure(figsize=(9,3), dpi=300)

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cam, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
