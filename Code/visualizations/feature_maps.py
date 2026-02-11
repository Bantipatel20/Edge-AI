import matplotlib.pyplot as plt
from .utils import ensure_dir

def save_feature_maps(feature_maps, out_dir, layer_name, max_maps=8):
    ensure_dir(out_dir)
    for i in range(min(max_maps, feature_maps.shape[1])):
        plt.figure(figsize=(3,3), dpi=300)
        plt.imshow(feature_maps[0, i], cmap="viridis")
        plt.axis("off")
        plt.title(f"{layer_name} | Ch {i}")
        plt.savefig(f"{out_dir}/{layer_name}_{i}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
