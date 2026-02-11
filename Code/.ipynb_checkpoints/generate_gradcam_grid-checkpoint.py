import os
from PIL import Image
import matplotlib.pyplot as plt

def generate_grid(original_dir, overlay_dir, output_pdf, limit=5):
    pairs = []
    for fname in sorted(os.listdir(original_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            original_path = os.path.join(original_dir, fname)
            overlay_path = os.path.join(overlay_dir, f"gradcam_{fname}")
            if os.path.exists(overlay_path):
                pairs.append((original_path, overlay_path))
    rows = min(len(pairs), limit)
    fig, axs = plt.subplots(rows, 2, figsize=(8, 4 * rows))
    for idx, (orig, over) in enumerate(pairs[:limit]):
        orig_img = Image.open(orig).resize((256, 256))
        over_img = Image.open(over).resize((256, 256))
        axs[idx, 0].imshow(orig_img)
        axs[idx, 0].set_title("Original")
        axs[idx, 0].axis("off")
        axs[idx, 1].imshow(over_img)
        axs[idx, 1].set_title("Grad-CAM Overlay")
        axs[idx, 1].axis("off")
    plt.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved visual grid to: {output_pdf}")

if __name__ == "__main__":
    original_dir = "./GradCAM_Overlays/Mixed"  # <-- Update as needed
    overlay_dir = "./GradCAM_Overlays/Mixed"
    output_pdf = "./plots/gradcam_visual_grid.pdf"
    generate_grid(original_dir, overlay_dir, output_pdf, limit=5)
