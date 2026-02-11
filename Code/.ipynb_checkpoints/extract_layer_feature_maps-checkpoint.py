import os
import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def convert_to_9_channels(img: Image.Image):
    rgb = np.array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    combined = np.concatenate([rgb, hsv, lab], axis=2)
    combined = combined.transpose(2, 0, 1) / 255.0
    return combined.astype(np.float32)

def modify_densenet_input_channels(model, in_channels=9):
    first_conv = model.features.conv0
    new_conv = nn.Conv2d(in_channels, first_conv.out_channels,
                         kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride,
                         padding=first_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = first_conv.weight
        for i in range(3, in_channels):
            new_conv.weight[:, i:i+1] = first_conv.weight[:, i % 3:i % 3 + 1]
    model.features.conv0 = new_conv
    return model

def save_feature_maps(name, feature_map, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    batch_size, num_channels, h, w = feature_map.shape
    for i in range(min(8, num_channels)):
        fmap = feature_map[0, i].detach().cpu().numpy()
        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        plt.title(f'{name} - Channel {i}')
        plt.savefig(f"{save_dir}/channel_{i}.png", bbox_inches='tight')
        plt.close()

def extract_and_save_feature_maps(img_path, model_path, output_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.densenet121(pretrained=False)
    model = modify_densenet_input_channels(model)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    activations = {}
    def get_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    layers_to_hook = {
        "conv0": model.features.conv0,
        "denseblock1": model.features.denseblock1,
        "denseblock2": model.features.denseblock2,
        "denseblock3": model.features.denseblock3,
        "denseblock4": model.features.denseblock4,
        "classifier": model.classifier
    }
    for name, layer in layers_to_hook.items():
        layer.register_forward_hook(get_hook(name))
    img = Image.open(img_path).convert('RGB').resize((512, 512))
    input_tensor = torch.tensor([convert_to_9_channels(img)]).to(device)
    _ = model(input_tensor)
    for name, fmap in activations.items():
        save_dir = os.path.join(output_root, name)
        save_feature_maps(name, fmap, save_dir)
    print(f"Feature maps saved in '{output_root}' folder.")

if __name__ == "__main__":
    img_path = "../YourDatasetSplits/Test/Mixed/20210927_135405.jpg"  # <-- Update as needed
    model_path = "./best_densenet_9ch.pth"
    output_root = "./layer_outputs"
    extract_and_save_feature_maps(img_path, model_path, output_root)
