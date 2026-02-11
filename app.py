import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import os

# ----------------------------------------------------------
# MODEL SETUP
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained DenseNet model with 9 channels"""
    def modify_densenet_input_channels(model, in_channels=9):
        conv = model.features.conv0
        new_conv = nn.Conv2d(
            in_channels, conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = conv.weight
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1] = conv.weight[:, i % 3:i % 3 + 1]

        model.features.conv0 = new_conv
        return model

    # Load DenseNet121 and modify for 9 channels
    model = models.densenet121(weights=None)
    model = modify_densenet_input_channels(model, in_channels=9)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    
    # Load trained weights
    model_path = "Code/best_densenet_9ch.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def preprocess_image(image):
    """Preprocess image to 9-channel format (RGB + HSV + LAB)"""
    # Resize image
    transform = transforms.Resize((512, 512))
    img = transform(image)
    img = np.array(img)
    
    # Convert RGB to HSV and LAB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    
    # Combine all channels (9 channels total)
    combined = np.concatenate([img, hsv, lab], axis=2)
    combined = combined.transpose(2, 0, 1) / 255.0
    
    # Convert to tensor
    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
    return tensor

def predict(model, image_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = output.argmax(1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, probabilities[0].numpy()

# ----------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Drug Classification System",
        page_icon="💊",
        layout="wide"
    )
    
    st.title("💊 Drug Classification System")
    st.markdown("---")
    st.write("Upload an image to classify whether it is **Mixed** or **Not Mixed**")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This application uses a DenseNet121 model with 9-channel input 
        (RGB + HSV + LAB color spaces) to classify drug images.
        
        **Classes:**
        - Mixed
        - Not Mixed
        """
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["png", "jpg", "jpeg"],
        help="Upload a drug image for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width='stretch')
        
        with col2:
            st.subheader("Classification Result")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                image_tensor = preprocess_image(image)
                prediction, confidence, probabilities = predict(model, image_tensor)
            
            # Display results
            class_names = ["Mixed", "Not_Mixed"]
            predicted_class = class_names[prediction]
            
            # Result with styled box
            if predicted_class == "Mixed":
                st.error(f"### 🔴 {predicted_class}")
            else:
                st.success(f"### 🟢 {predicted_class}")
            
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            
            # Probability bar chart
            st.write("**Class Probabilities:**")
            prob_dict = {
                "Mixed": probabilities[0] * 100,
                "Not Mixed": probabilities[1] * 100
            }
            st.bar_chart(prob_dict)
            
            # Detailed probabilities
            with st.expander("View Detailed Probabilities"):
                for i, class_name in enumerate(class_names):
                    st.write(f"**{class_name}:** {probabilities[i] * 100:.2f}%")
    
    else:
        st.info("👆 Please upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
