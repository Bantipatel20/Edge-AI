import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models

def convert_to_9_channels(img: Image.Image):
    rgb = np.array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    combined = np.concatenate([rgb, hsv, lab], axis=2)
    combined = combined.transpose(2, 0, 1) / 255.0
    return combined.astype(np.float32)

def modify_densenet_input_channels(model, in_channels=9):
    first_conv = model.features.conv0
    new_conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=first_conv.out_channels,
                               kernel_size=first_conv.kernel_size,
                               stride=first_conv.stride,
                               padding=first_conv.padding,
                               bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = first_conv.weight
        for i in range(3, in_channels):
            new_conv.weight[:, i:i+1] = first_conv.weight[:, i % 3:i % 3 + 1]
    model.features.conv0 = new_conv
    return model

def generate_gradcam(model, image_tensor, device):
    target_layer = model.features.norm5
    activations = {}
    gradients = {}
    def forward_hook(module, input, output):
        activations["value"] = output.detach()
    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_()
    output = model(image_tensor)
    pred_class = output.argmax().item()
    loss = output[0, pred_class]
    model.zero_grad()
    loss.backward()
    act = activations["value"][0]
    grad = gradients["value"][0]
    weights = grad.mean(dim=(1, 2))
    cam = (weights.view(-1, 1, 1) * act).sum(0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam.cpu().numpy(), pred_class

def overlay_cam_on_image(cam, original_img_tensor):
    cam = cv2.resize(cam, (original_img_tensor.shape[1], original_img_tensor.shape[2]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    rgb_img = original_img_tensor[:3].permute(1, 2, 0).cpu().numpy()
    rgb_img = np.uint8(rgb_img * 255)
    overlay = cv2.addWeighted(rgb_img, 0.5, heatmap, 0.5, 0)
    return overlay

def batch_generate_gradcam(input_folder, output_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=False)
    model = modify_densenet_input_channels(model, in_channels=9)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    for fname in os.listdir(input_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, fname)
            img = Image.open(img_path).convert('RGB').resize((512, 512))
            img_tensor = torch.tensor(convert_to_9_channels(img))
            cam, pred = generate_gradcam(model, img_tensor, device)
            overlay = overlay_cam_on_image(cam, img_tensor)
            out_path = os.path.join(output_folder, f"gradcam_{fname}")
            cv2.imwrite(out_path, overlay[:, :, ::-1])
            print(f"Saved Grad-CAM overlay for {fname} to {out_path}")

if __name__ == "__main__":
    input_folder = "../YourDatasetSplits/Test/Mixed"  # <-- Update as needed
    output_folder = "./GradCAM_Overlays/Mixed"
    model_path = "./best_densenet_9ch.pth"
    os.makedirs(output_folder, exist_ok=True)
    batch_generate_gradcam(input_folder, output_folder, model_path)
