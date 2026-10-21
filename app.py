import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models


INPUT_SIZE = (512, 512)
CLASS_NAMES = ["Mixed", "Not Mixed"]


# ----------------------------------------------------------
# MODEL SETUP
# ----------------------------------------------------------
def get_last_conv_layer(model):
    """Return the last convolutional layer used for Grad-CAM."""
    last_conv = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


@st.cache_resource
def load_model():
    """Load the trained 9-channel DenseNet121 model and its Grad-CAM layer."""
    def modify_densenet_input_channels(model, in_channels=9):
        conv = model.features.conv0
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = conv.weight
            for i in range(3, in_channels):
                new_conv.weight[:, i : i + 1] = conv.weight[:, i % 3 : i % 3 + 1]

        model.features.conv0 = new_conv
        return model

    model = models.densenet121(weights=None)
    model = modify_densenet_input_channels(model, in_channels=9)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "Code/best_densenet_9ch.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")

    return model, target_layer, device


def preprocess_image(image):
    """Convert an RGB image to the model's 9-channel RGB + HSV + LAB input."""
    resized = transforms.Resize(INPUT_SIZE)(image)
    rgb = np.array(resized)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    combined = np.concatenate([rgb, hsv, lab], axis=2)
    combined = combined.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(combined).unsqueeze(0)


def predict(model, image_tensor, device):
    """Run inference and return the predicted class and class probabilities."""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = output.argmax(1).item()
        confidence = probabilities[0, prediction].item()

    return prediction, confidence, probabilities[0].detach().cpu().numpy()


def generate_gradcam_overlay(model, target_layer, image_tensor, original_rgb, class_index, device):
    """Generate a Grad-CAM overlay and its raw heatmap for a selected class."""
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(image_tensor.to(device))
            output[:, class_index].sum().backward()

        if not activations or not gradients:
            return original_rgb, original_rgb

        activation_map = activations[0]
        gradient_map = gradients[0]
        weights = gradient_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation_map).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        height, width = original_rgb.shape[:2]
        cam = cv2.resize(cam, (width, height))
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_rgb, 0.62, heatmap, 0.38, 0)
        return overlay, heatmap
    finally:
        forward_handle.remove()
        backward_handle.remove()


# ----------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------
def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --ink: #17233b;
                --muted: #68758b;
                --blue: #356df3;
                --teal: #15b8a6;
                --surface: #ffffff;
                --line: #e6ebf3;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 0%, rgba(53,109,243,.08), transparent 28%),
                    radial-gradient(circle at 95% 8%, rgba(21,184,166,.07), transparent 24%),
                    #f5f7fb;
                color: var(--ink);
            }
            .block-container {
                max-width: 1280px;
                padding-top: 1.8rem !important;
                padding-bottom: 2rem !important;
            }
            [data-testid="stSidebar"] { display: none; }
            header[data-testid="stHeader"] { background: transparent; }
            [data-testid="stToolbar"], .stDeployButton, #MainMenu { display: none; }

            .hero {
                position: relative;
                overflow: hidden;
                padding: 2rem 2.2rem;
                margin-bottom: 1.5rem;
                border-radius: 24px;
                background: linear-gradient(120deg, #14264b 0%, #213d79 58%, #2867ae 100%);
                box-shadow: 0 18px 45px rgba(28, 55, 105, .19);
            }
            .hero:after {
                content: "";
                position: absolute;
                width: 280px;
                height: 280px;
                right: -70px;
                top: -120px;
                border-radius: 50%;
                border: 1px solid rgba(255,255,255,.16);
                box-shadow: 0 0 0 32px rgba(255,255,255,.035), 0 0 0 68px rgba(255,255,255,.025);
            }
            .hero-kicker {
                display: inline-block;
                color: #b9e8ff;
                background: rgba(255,255,255,.11);
                border: 1px solid rgba(255,255,255,.15);
                padding: .32rem .72rem;
                border-radius: 999px;
                text-transform: uppercase;
                letter-spacing: .13em;
                font-size: .7rem;
                font-weight: 700;
            }
            .hero h1 {
                position: relative;
                z-index: 1;
                color: #ffffff;
                font-size: clamp(2rem, 4vw, 3.15rem);
                line-height: 1.08;
                margin: .85rem 0 .55rem;
                letter-spacing: -.04em;
                font-weight: 800;
            }
            .hero p {
                position: relative;
                z-index: 1;
                color: #d5e3ff;
                max-width: 690px;
                font-size: 1rem;
                margin: 0;
                line-height: 1.65;
            }
            .hero-tags { margin-top: 1.15rem; display: flex; gap: .5rem; flex-wrap: wrap; }
            .hero-tag {
                color: #f0f6ff;
                border: 1px solid rgba(255,255,255,.2);
                background: rgba(255,255,255,.075);
                border-radius: 8px;
                padding: .33rem .6rem;
                font-size: .75rem;
            }

            .section-heading { margin: 1.8rem 0 .25rem; }
            .section-eyebrow {
                color: var(--blue);
                text-transform: uppercase;
                letter-spacing: .12em;
                font-weight: 800;
                font-size: .69rem;
                margin-bottom: .28rem;
            }
            .section-title {
                color: var(--ink);
                font-size: 1.35rem;
                letter-spacing: -.025em;
                font-weight: 750;
                margin: 0;
            }
            .section-subtitle { color: var(--muted); margin: .3rem 0 0; font-size: .91rem; }

            .panel {
                background: rgba(255,255,255,.94);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1.15rem 1.25rem;
                box-shadow: 0 5px 20px rgba(25, 43, 77, .04);
            }
            .mini-label {
                color: var(--muted);
                font-size: .72rem;
                letter-spacing: .09em;
                text-transform: uppercase;
                font-weight: 800;
                margin-bottom: .5rem;
            }
            .how-step { display: flex; align-items: flex-start; gap: .7rem; margin: .65rem 0; }
            .step-number {
                min-width: 1.65rem;
                height: 1.65rem;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 9px;
                color: #275ce0;
                background: #edf3ff;
                font-size: .78rem;
                font-weight: 800;
            }
            .step-copy { color: #53627a; font-size: .86rem; line-height: 1.45; }
            .step-copy strong { color: var(--ink); }

            /* File uploader: enforce readable light colors across Streamlit themes/versions. */
            div[data-testid="stFileUploader"] {
                background: #ffffff !important;
                color: #24324a !important;
                border: 1.5px dashed #cbd7ec;
                border-radius: 14px;
                padding: .65rem;
            }
            div[data-testid="stFileUploader"] label,
            div[data-testid="stFileUploader"] label * ,
            div[data-testid="stFileUploader"] p,
            div[data-testid="stFileUploader"] small,
            div[data-testid="stFileUploader"] span {
                color: #43516a !important;
            }
            div[data-testid="stFileUploader"] section,
            div[data-testid="stFileUploaderDropzone"] {
                background: #f7f9fe !important;
                color: #24324a !important;
                border: 1px dashed #cbd7ec !important;
                border-radius: 11px !important;
            }
            div[data-testid="stFileUploaderDropzone"] * {
                color: #24324a !important;
            }
            div[data-testid="stFileUploaderDropzone"] button {
                background: #edf3ff !important;
                color: #244fb8 !important;
                border: 1px solid #c8d7ff !important;
                border-radius: 9px !important;
            }
            div[data-testid="stFileUploaderDropzone"] button:hover {
                background: #e1ebff !important;
                color: #193f9e !important;
                border-color: #a9c0ff !important;
            }
            div[data-testid="stFileUploader"] section { border: none; }
            div[data-testid="stCameraInput"] { border-radius: 14px; overflow: hidden; }
            div[role="radiogroup"] { gap: .65rem; }
            div[role="radiogroup"] label {
                background: #fff !important;
                color: #24324a !important;
                padding: .48rem .8rem;
                border: 1px solid var(--line);
                border-radius: 10px;
            }
            div[role="radiogroup"] label * {
                color: #24324a !important;
            }
            .stButton > button {
                border-radius: 10px;
                border: 0;
                background: linear-gradient(100deg, #356df3, #2856cc);
                color: #fff;
                font-weight: 700;
                padding: .55rem 1rem;
            }
            .stButton > button:hover { color: white; border: 0; filter: brightness(1.06); }

            .result-banner {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                padding: 1.2rem 1.35rem;
                border-radius: 17px;
                border: 1px solid;
                margin: .8rem 0 1rem;
            }
            .result-banner.mixed { background: #fff3f2; border-color: #ffd4d0; }
            .result-banner.not-mixed { background: #edfbf5; border-color: #bcebd7; }
            .result-label {
                font-size: .7rem;
                letter-spacing: .12em;
                text-transform: uppercase;
                font-weight: 800;
                margin-bottom: .3rem;
            }
            .mixed .result-label, .mixed .result-name { color: #b83232; }
            .not-mixed .result-label, .not-mixed .result-name { color: #087f5b; }
            .result-name { font-size: 1.45rem; font-weight: 850; letter-spacing: -.025em; }
            .result-description { color: #57657b; margin-top: .2rem; font-size: .88rem; }
            .result-confidence { text-align: right; min-width: 100px; }
            .result-confidence strong { display: block; font-size: 1.8rem; line-height: 1.15; color: var(--ink); }
            .result-confidence span { color: var(--muted); font-size: .72rem; }

            [data-testid="stMetric"] {
                background: #fff;
                border: 1px solid var(--line);
                padding: 1rem 1.1rem;
                border-radius: 15px;
                box-shadow: 0 5px 18px rgba(25,43,77,.035);
            }
            [data-testid="stMetricLabel"] { color: #6b7890; }
            [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.55rem; font-weight: 800; }
            [data-testid="stImage"] img { border-radius: 13px; }
            .image-caption {
                color: #65728a;
                font-size: .82rem;
                padding-top: .35rem;
            }
            .probability-label {
                display: flex;
                justify-content: space-between;
                font-size: .87rem;
                font-weight: 700;
                color: #293852;
                margin: .85rem 0 .35rem;
            }
            .footnote {
                color: #7b8799;
                font-size: .76rem;
                text-align: center;
                padding: 1.3rem .5rem .1rem;
            }
            .stProgress > div > div > div > div { background: linear-gradient(90deg, #356df3, #15b8a6); }
            .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
            .stTabs [data-baseweb="tab"] {
                height: 2.8rem;
                background: #eef2f9;
                border-radius: 10px 10px 0 0;
                padding: 0 1rem;
            }
            .stTabs [aria-selected="true"] { background: #fff !important; color: #285ce0 !important; }
            .stAlert { border-radius: 12px; }

            @media (max-width: 700px) {
                .block-container { padding-left: 1rem; padding-right: 1rem; }
                .hero { padding: 1.5rem; border-radius: 18px; }
                .result-banner { align-items: flex-start; flex-direction: column; }
                .result-confidence { text-align: left; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_heading(eyebrow, title, subtitle=None):
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-heading"><div class="section-eyebrow">{eyebrow}</div>'
        f'<h2 class="section-title">{title}</h2>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="EDGE AI | Drug Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <span class="hero-kicker">Computer vision · AI analysis</span>
            <h1>EDGE AI</h1>
            <p>Drug Classification Studio — analyze an image, review the model prediction, and explore visual explanations powered by DenseNet121.</p>
            <div class="hero-tags">
                <span class="hero-tag">DenseNet121</span>
                <span class="hero-tag">Grad-CAM explainability</span>
                <span class="hero-tag">RGB + HSV + LAB</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading(
        "01 · Start here",
        "Add an image to analyze",
        "Upload a clear JPG or PNG, or use your camera to capture an image.",
    )

    upload_col, guide_col = st.columns([1.55, 1], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown('<div class="mini-label">Choose input source</div>', unsafe_allow_html=True)
            input_mode = st.radio(
                "Input method",
                options=["Upload image", "Use camera"],
                horizontal=True,
                label_visibility="collapsed",
                key="input_mode",
            )
            if input_mode == "Upload image":
                input_file = st.file_uploader(
                    "Drop an image here or browse files",
                    type=["png", "jpg", "jpeg"],
                    help="Supported formats: PNG, JPG, and JPEG.",
                    label_visibility="visible",
                )
            else:
                input_file = st.camera_input("Capture an image")
    with guide_col:
        st.markdown(
            """
            <div class="panel">
                <div class="mini-label">What happens next</div>
                <div class="how-step"><div class="step-number">1</div><div class="step-copy"><strong>Image processing</strong><br>Color information is prepared for the model.</div></div>
                <div class="how-step"><div class="step-number">2</div><div class="step-copy"><strong>Classification</strong><br>The model predicts Mixed or Not Mixed.</div></div>
                <div class="how-step"><div class="step-number">3</div><div class="step-copy"><strong>Visual explanation</strong><br>Grad-CAM highlights regions influencing the prediction.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if input_file is not None:
        try:
            image = Image.open(input_file).convert("RGB")
        except Exception as exc:
            st.error(f"Unable to open this image. Please try another file. Details: {exc}")
            return

        original_rgb = np.array(image)
        with st.spinner("Analyzing image and preparing visual explanations…"):
            try:
                model, target_layer, device = load_model()
                image_tensor = preprocess_image(image)
                prediction, confidence, probabilities = predict(model, image_tensor, device)

                gradcam_overlay, _ = generate_gradcam_overlay(
                    model, target_layer, image_tensor, original_rgb, prediction, device
                )
                gradcam_mixed, heatmap_mixed = generate_gradcam_overlay(
                    model, target_layer, image_tensor, original_rgb, 0, device
                )
                gradcam_notmixed, heatmap_notmixed = generate_gradcam_overlay(
                    model, target_layer, image_tensor, original_rgb, 1, device
                )
            except Exception as exc:
                st.error(
                    "The image could not be analyzed. Verify that the model weights are available at "
                    "`Code/best_densenet_9ch.pth` and that the required packages are installed."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))
                return

        predicted_class = CLASS_NAMES[prediction]
        is_mixed = predicted_class == "Mixed"

        section_heading("02 · Model output", "Classification result", "Prediction and confidence for the selected image.")
        status_class = "mixed" if is_mixed else "not-mixed"
        result_description = "The model classified this image as Mixed." if is_mixed else "The model classified this image as Not Mixed."
        st.markdown(
            f'<div class="result-banner {status_class}">'
            f'<div><div class="result-label">Prediction complete</div>'
            f'<div class="result-name">{"● MIXED" if is_mixed else "● NOT MIXED"}</div>'
            f'<div class="result-description">{result_description}</div></div>'
            f'<div class="result-confidence"><strong>{confidence * 100:.1f}%</strong><span>model confidence</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3, gap="medium")
        with metric_col1:
            st.metric("Predicted class", predicted_class)
        with metric_col2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        with metric_col3:
            st.metric("Classes evaluated", "2")

        section_heading("03 · Visual explanation", "See what the model focused on", "Grad-CAM overlays help interpret the model output; they are not a guarantee of correctness.")
        original_col, overlay_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("**Input image**")
            st.image(original_rgb, use_container_width=True)
            st.caption("Original image supplied for analysis")
        with overlay_col:
            st.markdown(f"**Prediction attention · {predicted_class}**")
            st.image(gradcam_overlay, use_container_width=True)
            st.caption("Grad-CAM overlay for the predicted class")

        with st.expander("Compare both classes and raw heatmaps", expanded=False):
            compare_tab, raw_tab = st.tabs(["Grad-CAM comparison", "Raw heatmaps"])
            with compare_tab:
                mixed_col, notmixed_col = st.columns(2, gap="large")
                with mixed_col:
                    st.markdown("**Mixed class attention**")
                    st.image(gradcam_mixed, use_container_width=True)
                with notmixed_col:
                    st.markdown("**Not Mixed class attention**")
                    st.image(gradcam_notmixed, use_container_width=True)
            with raw_tab:
                raw_mixed_col, raw_notmixed_col = st.columns(2, gap="large")
                with raw_mixed_col:
                    st.markdown("**Mixed heatmap**")
                    st.image(heatmap_mixed, use_container_width=True)
                with raw_notmixed_col:
                    st.markdown("**Not Mixed heatmap**")
                    st.image(heatmap_notmixed, use_container_width=True)

        section_heading("04 · Confidence breakdown", "Class probabilities", "The bars show the model's probability for each available class.")
        probability_col, probability_note_col = st.columns([1.25, 1], gap="large")
        with probability_col:
            for class_name, probability in zip(CLASS_NAMES, probabilities):
                percentage = float(probability) * 100
                st.markdown(
                    f'<div class="probability-label"><span>{class_name}</span><span>{percentage:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(float(probability), 0.0), 1.0))
        with probability_note_col:
            st.markdown(
                """
                <div class="panel">
                    <div class="mini-label">How to read this</div>
                    <div class="step-copy">A higher probability means the model favored that class more strongly for this input. Confidence reflects the model's output, not laboratory verification.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info("This result is an AI classification output for review and should not replace laboratory testing or expert assessment.")
    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding:2.3rem 1.2rem; margin-top:1.2rem;">
                <div style="font-size:2rem; margin-bottom:.35rem;">✦</div>
                <div style="font-size:1.15rem; font-weight:800; color:#17233b;">Your analysis workspace is ready</div>
                <div style="color:#68758b; margin-top:.45rem;">Choose an image above to see the prediction and visual explanation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footnote">EDGE AI · DenseNet121 · Grad-CAM explainability</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
