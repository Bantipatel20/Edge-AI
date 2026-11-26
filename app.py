"""
EDGE AI - Drug Classification (DenseNet121, 9-channel RGB+HSV+Lab, Grad-CAM)

Same model, preprocessing and class order as the training pipeline:
Mixed = 0, Not Mixed = 1.
"""
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

INPUT_SIZE = (512, 512)
CLASS_NAMES = ["Mixed", "Not Mixed"]
CLASS_NOTE = ["Drug mixture detected", "Single drug detected"]

APP_DIR = Path(__file__).resolve().parent


def first_existing(candidates, default):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path(default)


MODEL_PATH = first_existing(
    [os.environ.get("MODEL_PATH"), APP_DIR / "Code" / "best_densenet_9ch.pth", APP_DIR / "best_densenet_9ch.pth"],
    APP_DIR / "Code" / "best_densenet_9ch.pth",
)
PLOTS_DIR = first_existing(
    [os.environ.get("PLOTS_DIR"), APP_DIR / "assets" / "plots", APP_DIR / "Code" / "Code" / "plots",
     APP_DIR / "Code" / "plots", APP_DIR / "plots"],
    APP_DIR / "assets" / "plots",
)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def get_last_conv_layer(model):
    last = None
    for m in model.features.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    def widen_input(model, in_channels=9):
        conv = model.features.conv0
        new_conv = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                             stride=conv.stride, padding=conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = conv.weight
            for i in range(3, in_channels):
                new_conv.weight[:, i:i + 1] = conv.weight[:, i % 3:i % 3 + 1]
        model.features.conv0 = new_conv
        return model

    model = models.densenet121(weights=None)
    model = widen_input(model, 9)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    target = get_last_conv_layer(model)
    if target is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return model, target, device


def preprocess_image(image):
    img = np.array(transforms.Resize(INPUT_SIZE)(image))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    combined = np.concatenate([img, hsv, lab], axis=2).transpose(2, 0, 1) / 255.0
    return torch.tensor(combined, dtype=torch.float32).unsqueeze(0)


def run_model(model, target_layer, tensor, device, num_features=16):
    """One forward pass gives probabilities, Grad-CAM for both classes and the
    first-layer feature maps."""
    acts, grads, fmaps = [], [], []
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    h3 = model.features.conv0.register_forward_hook(lambda m, i, o: fmaps.append(o.detach()))
    try:
        x = tensor.to(device)
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            out = model(x)
            probs = F.softmax(out, dim=1)[0].detach().cpu().numpy()
            for cls in (0, 1):
                out[:, cls].sum().backward(retain_graph=(cls == 0))
    finally:
        h1.remove(); h2.remove(); h3.remove()

    # backward for class 0 ran first, class 1 second
    cams = []
    act = acts[0].detach()
    for g in grads[:2]:
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * act).sum(dim=1)).squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        cams.append(cam.astype(np.float32))

    fm = fmaps[0][0].cpu().numpy()[:num_features]
    return probs, int(probs.argmax()), cams, fm


def heatmap_rgb(cam, size):
    cam = cv2.resize(cam, size)
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(rgb, cam, strength):
    heat = heatmap_rgb(cam, (rgb.shape[1], rgb.shape[0]))
    return cv2.addWeighted(rgb, 1 - strength, heat, strength, 0)


def feature_grid(fmaps, cols=4):
    tiles = []
    for fm in fmaps:
        n = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        tiles.append(cv2.cvtColor(cv2.applyColorMap(np.uint8(n * 255), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB))
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return grid


@st.cache_data(show_spinner=False, max_entries=6)
def analyze(_model, _layer, _device, data: bytes):
    """Cached per image, so changing a slider or tab never re-runs the model."""
    t0 = time.perf_counter()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    probs, pred, cams, fm = run_model(_model, _layer, preprocess_image(image), _device)
    ms = (time.perf_counter() - t0) * 1000
    rgb = np.array(image)
    scale = 900 / max(rgb.shape[:2])
    if scale < 1:  # keep the page light for large photos
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return {"rgb": rgb, "probs": probs, "pred": pred, "cams": cams, "fm": fm, "ms": ms}


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');
:root{--paper:#F2F5F3;--panel:#fff;--ink:#10242A;--muted:#5B6F73;--line:#D5DFDC;
  --teal:#0B6E75;--teal-dark:#0A4F55;--a:#0B6E75;--a-bg:#E1F1F2;--b:#B45309;--b-bg:#FBEBD6}
html,body,[class*="css"],.stApp{font-family:'Public Sans','Segoe UI',system-ui,sans-serif;color:var(--ink)}
.stApp{background:var(--paper)}
h1,h2,h3,h4{font-family:'Bricolage Grotesque','Segoe UI',system-ui,sans-serif !important;letter-spacing:-.01em;color:var(--ink)}
#MainMenu,footer,[data-testid="stDecoration"],.stDeployButton,[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none !important}
header[data-testid="stHeader"]{display:none}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1240px}

.topbar{background:#0D2428;border-radius:18px;padding:1.3rem 1.6rem;display:flex;align-items:center;
  justify-content:space-between;gap:1rem;margin-bottom:1.4rem}
.brand{display:flex;align-items:center;gap:.9rem}
.mark{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;font-size:1.4rem;
  background:linear-gradient(135deg,#17A2A9,#0B6E75)}
.bname{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.45rem;color:#fff;line-height:1.1}
.bsub{font-size:.85rem;color:#8FA9A8}
.chips{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:flex-end}
.chip{font-size:.78rem;color:#CFE3E1;background:#16353A;border:1px solid #21424A;border-radius:99px;padding:.3rem .75rem}

.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:2.7rem;padding:0 1.05rem;font-weight:600}
.stTabs [aria-selected="true"]{color:var(--teal-dark)}
.stTabs [data-baseweb="tab-highlight"]{background:var(--teal);height:3px}

.title h2{font-size:1.9rem;margin:.3rem 0 .2rem;font-weight:700}
.title p{color:var(--muted);margin:0 0 1.2rem;max-width:64ch}
.ptitle{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.05rem;margin:0 0 .1rem}
.psub{color:var(--muted);font-size:.86rem;margin:0 0 .8rem}

[data-testid="stFileUploaderDropzone"]{background:#F8FBFA;border:2px dashed #9DB8B5;border-radius:14px;padding:1.5rem 1rem}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--teal);background:#EEF7F6}
[data-testid="stFileUploaderDropzone"] button{border-radius:9px;border:1px solid var(--teal);color:var(--teal);background:#fff;font-weight:600}
[data-testid="stFileUploaderDropzone"] button:hover{background:var(--teal);color:#fff}
[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--line)}
[data-testid="stImageCaption"]{color:var(--muted);font-size:.82rem}

.verdict{border-radius:18px;padding:1.4rem 1.6rem 1.3rem;border:1px solid}
.verdict.c0{background:var(--a-bg);border-color:#B5D9DC}
.verdict.c1{background:var(--b-bg);border-color:#EBC999}
.vl{font-size:.85rem;font-weight:600;color:var(--muted);margin-bottom:.15rem}
.vm{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:3rem;line-height:1.02;letter-spacing:-.02em}
.verdict.c0 .vm{color:var(--a)} .verdict.c1 .vm{color:var(--b)}
.vn{margin-top:.5rem;font-size:.97rem;color:#2C4045}

.bars{margin-top:1.2rem;display:grid;gap:.75rem}
.bar-row{display:grid;grid-template-columns:88px 1fr 56px;align-items:center;gap:.7rem;font-size:.88rem;font-weight:600}
.track{height:12px;border-radius:99px;background:rgba(16,36,42,.1);overflow:hidden}
.fill{height:100%;border-radius:99px;transition:width .5s cubic-bezier(.2,.8,.2,1)}
.fill.c0{background:var(--a)} .fill.c1{background:var(--b)}
.bar-row span:last-child{text-align:right}

.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:.9rem}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem}
.sk{font-size:.78rem;color:var(--muted)} .sv{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.45rem}

.empty{border:1px dashed var(--line);border-radius:16px;padding:2.6rem 1.5rem;text-align:center;background:rgba(255,255,255,.55)}
.empty h3{margin:.3rem 0;font-size:1.25rem} .empty p{color:var(--muted);margin:0 auto;max-width:48ch}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.3rem}
.kv{display:grid;grid-template-columns:150px 1fr;gap:.55rem 1rem;font-size:.95rem}
.kv div:nth-child(odd){color:var(--muted);font-weight:500}
.foot{text-align:center;color:var(--muted);font-size:.82rem;margin-top:2.2rem}

.stButton>button{border-radius:10px;font-weight:600;border:1px solid var(--teal);background:var(--teal);color:#fff;padding:.45rem 1.1rem}
.stDownloadButton>button{border-radius:10px;font-weight:600;border:1px solid var(--teal);background:#fff;color:var(--teal)}
.stDownloadButton>button:hover{background:var(--teal);color:#fff;border-color:var(--teal)}
.stButton>button:hover{background:var(--teal-dark);border-color:var(--teal-dark);color:#fff}
button:focus-visible{outline:3px solid #7FD3D8;outline-offset:2px}
@media (max-width:900px){.stats{grid-template-columns:1fr}.vm{font-size:2.3rem}.topbar{flex-direction:column;align-items:flex-start}.chips{justify-content:flex-start}}
@media (prefers-reduced-motion:reduce){*{transition:none !important}}
</style>
"""


def verdict_html(pred, probs):
    bars = "".join(
        f'<div class="bar-row"><span>{CLASS_NAMES[i]}</span>'
        f'<div class="track"><div class="fill c{i}" style="width:{probs[i]*100:.1f}%"></div></div>'
        f'<span>{probs[i]*100:.1f}%</span></div>' for i in (0, 1)
    )
    return (f'<div class="verdict c{pred}"><div class="vl">Prediction</div>'
            f'<div class="vm">{CLASS_NAMES[pred]}</div><div class="vn">{CLASS_NOTE[pred]}</div>'
            f'<div class="bars">{bars}</div></div>')


def stats_html(conf, ms, device):
    return ('<div class="stats">'
            f'<div class="stat"><div class="sk">Confidence</div><div class="sv">{conf:.1f}%</div></div>'
            f'<div class="stat"><div class="sk">Analysis time</div><div class="sv">{ms/1000:.2f} s</div></div>'
            f'<div class="stat"><div class="sk">Running on</div><div class="sv">{device.type.upper()}</div></div></div>')


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="EDGE AI | Drug Classification", page_icon="🧠",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """<div class="topbar"><div class="brand"><div class="mark">🧠</div>
        <div><div class="bname">EDGE AI</div><div class="bsub">Drug classification system</div></div></div>
        <div class="chips"><span class="chip">DenseNet121</span><span class="chip">9-channel input</span>
        <span class="chip">Grad-CAM</span><span class="chip">Runs on the edge</span></div></div>""",
        unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error(f"Model file not found at `{MODEL_PATH}`. Put `best_densenet_9ch.pth` there "
                 "or set the MODEL_PATH environment variable.")
        st.stop()

    with st.spinner("Loading model…"):
        model, target_layer, device = load_model(str(MODEL_PATH))

    tab_inspect, tab_perf, tab_about = st.tabs(["Classify", "Model performance", "About"])

    # ------------------------------------------------------------ Classify
    with tab_inspect:
        st.markdown('<div class="title"><h2>Classify a sample</h2><p>Add an image of the sample. '
                    'The model returns a class and shows which regions drove the decision.</p></div>',
                    unsafe_allow_html=True)

        left, right = st.columns([5, 7], gap="large")
        with left:
            st.markdown('<div class="ptitle">Input</div><div class="psub">PNG or JPG. Resized to 512 × 512 for the model.</div>',
                        unsafe_allow_html=True)
            mode = st.radio("Input method", ["Upload image", "Camera"], horizontal=True, label_visibility="collapsed")
            if mode == "Upload image":
                src = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            else:
                src = st.camera_input("Capture an image", label_visibility="collapsed")
            strength = st.slider("Heatmap strength", 0.2, 0.8, 0.4, 0.05,
                                 help="How strongly the Grad-CAM colours blend over the image.")

        if src is None:
            with right:
                st.markdown('<div class="empty"><div style="font-size:2rem">🧪</div><h3>Ready for analysis</h3>'
                            '<p>Upload an image or take a photo. The class, confidence and heatmaps appear here.</p></div>',
                            unsafe_allow_html=True)
        else:
            data = src.getvalue()
            with st.spinner("Analysing image…"):
                res = analyze(model, target_layer, device, data)
            rgb, probs, pred, cams = res["rgb"], res["probs"], res["pred"], res["cams"]
            size = (rgb.shape[1], rgb.shape[0])

            with right:
                st.markdown(verdict_html(pred, probs), unsafe_allow_html=True)
                st.markdown(stats_html(probs[pred] * 100, res["ms"], device), unsafe_allow_html=True)

            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
            t1, t2, t3, t4 = st.tabs(["Predicted class", "Both classes", "Raw heatmaps", "Feature maps"])
            with t1:
                a, b = st.columns(2, gap="large")
                a.image(rgb, caption="Input image", use_container_width=True)
                cam_img = overlay(rgb, cams[pred], strength)
                b.image(cam_img, caption=f"Grad-CAM for {CLASS_NAMES[pred]}: warm areas mattered most",
                        use_container_width=True)
                ok, png = cv2.imencode(".png", cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
                if ok:
                    b.download_button("Download heatmap", png.tobytes(),
                                      file_name="gradcam.png", mime="image/png")
            with t2:
                a, b = st.columns(2, gap="large")
                a.image(overlay(rgb, cams[0], strength), caption="Where the model looked for Mixed", use_container_width=True)
                b.image(overlay(rgb, cams[1], strength), caption="Where the model looked for Not Mixed", use_container_width=True)
            with t3:
                a, b = st.columns(2, gap="large")
                a.image(heatmap_rgb(cams[0], size), caption="Mixed heatmap (no overlay)", use_container_width=True)
                b.image(heatmap_rgb(cams[1], size), caption="Not Mixed heatmap (no overlay)", use_container_width=True)
            with t4:
                st.image(feature_grid(res["fm"]), use_container_width=True,
                         caption="First 16 feature maps from the first convolutional layer")

    # ------------------------------------------------------------ Performance
    with tab_perf:
        st.markdown('<div class="title"><h2>Model performance</h2><p>Charts produced by the training pipeline. '
                    'Re-run it to refresh them.</p></div>', unsafe_allow_html=True)

        def plot(name, caption):
            p = PLOTS_DIR / f"{name}.png"
            if p.exists():
                st.image(str(p), caption=caption, use_container_width=True)
            else:
                st.info(f"{name}.png not found in {PLOTS_DIR}")

        p1, p2, p3, p4 = st.tabs(["Training", "Classification", "Curves", "Data and errors"])
        with p1:
            a, b = st.columns(2, gap="large")
            with a: plot("train_val_accuracy", "Training vs validation accuracy")
            with b: plot("train_val_loss", "Training vs validation loss")
        with p2:
            a, b = st.columns(2, gap="large")
            with a: plot("confusion_matrix", "Confusion matrix (counts)")
            with b: plot("confusion_matrix_normalized", "Confusion matrix (normalised)")
            a, b = st.columns(2, gap="large")
            with a: plot("classwise_metrics", "Precision, recall and F1 per class")
            with b: plot("final_validation_metrics", "Final validation metrics")
        with p3:
            a, b = st.columns(2, gap="large")
            with a: plot("roc_curve", "ROC curve")
            with b: plot("precision_recall_curve", "Precision-recall curve")
        with p4:
            a, b = st.columns([2, 3], gap="large")
            with a: plot("class_distribution", "Training class distribution")
            with b: plot("misclassified_samples", "Samples the model got wrong")

    # ------------------------------------------------------------ About
    with tab_about:
        st.markdown('<div class="title"><h2>About</h2><p>An edge-deployed classifier for drug powder images.</p></div>',
                    unsafe_allow_html=True)
        st.markdown(
            """<div class="panel"><div class="kv">
            <div>Task</div><div>Binary classification: Mixed or Not Mixed</div>
            <div>Model</div><div>DenseNet121, first layer widened to 9 channels</div>
            <div>Input</div><div>RGB + HSV + Lab stacked, 512 × 512 px</div>
            <div>Explainability</div><div>Grad-CAM on the last convolutional layer, for both classes</div>
            <div>Deployment</div><div>Runs on the edge device. Images stay on the machine.</div>
            </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="foot">EDGE AI · Built for edge inference · DenseNet121 + Grad-CAM</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
