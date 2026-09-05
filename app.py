import os
import tempfile
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="STENOSIS // SCAN",
    page_icon="📡",
    layout="wide"
)

# -----------------------------
# Custom CSS — instrument panel / scanner aesthetic
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

footer {visibility: hidden;}
.stApp {
    background-color: #0A0A0C;
}

/* Sidebar = control panel */
[data-testid="stSidebar"] {
    background-color: #111114;
    border-right: 1px solid #2A2A2E;
}
[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', monospace;
}

.panel-label {
    color: #6B6B70;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid #2A2A2E;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
}

/* Header strip */
.scan-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.6rem 1.5rem;
    border-bottom: 2px solid #E63946;
    padding-bottom: 0.8rem;
    margin-bottom: 1.6rem;
}
.scan-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 800;
    color: #F2F2F2;
    letter-spacing: -1px;
    white-space: nowrap;
}
.scan-title span {
    color: #E63946;
}
.scan-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #6B6B70;
    letter-spacing: 1px;
}

/* Readout blocks — no rounded corners, hairline borders */
.readout {
    border: 1px solid #2A2A2E;
    border-left: 3px solid #E63946;
    background-color: #111114;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
}
.readout-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #6B6B70;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.4rem;
}
.readout-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    color: #F2F2F2;
    font-weight: 700;
}
.readout-value.alert {
    color: #E63946;
}

.coord-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0;
    border: 1px solid #2A2A2E;
    margin-bottom: 1.2rem;
}
.coord-cell {
    padding: 0.8rem 1rem;
    border-right: 1px solid #2A2A2E;
    background-color: #111114;
}
.coord-cell:last-child { border-right: none; }
.coord-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #6B6B70;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.coord-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    color: #E63946;
    font-weight: 700;
}

.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #E63946;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.6rem;
    margin-top: 1rem;
}
.section-tag::before {
    content: "// ";
    color: #6B6B70;
}

[data-testid="stFileUploader"] {
    background-color: #111114;
    border: 1px dashed #2A2A2E;
    border-radius: 0;
}

div[data-testid="stImage"] img,
video {
    border: 1px solid #2A2A2E;
}

.stButton button, .stDownloadButton button {
    font-family: 'JetBrains Mono', monospace;
    background-color: #E63946;
    color: #0A0A0C;
    border: none;
    border-radius: 0;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stRadio [role="radiogroup"] label {
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "stenosis_detector_v18.keras"
IMG_SIZE = 512

STABLE_THRESHOLD = 2
IOU_THRESHOLD = 0.25

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Huber": tf.keras.losses.Huber}
    )

model = load_model()

# -----------------------------
# Prediction function
# -----------------------------
def predict_bbox(image):
    h, w, _ = image.shape

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_resized, verbose=0)
    ymin, xmin, ymax, xmax = preds[1][0]

    xmin = int(xmin / IMG_SIZE * w)
    xmax = int(xmax / IMG_SIZE * w)
    ymin = int(ymin / IMG_SIZE * h)
    ymax = int(ymax / IMG_SIZE * h)

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    return xmin, ymin, xmax, ymax


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    xi1 = max(x1, x1p)
    yi1 = max(y1, y1p)
    xi2 = min(x2, x2p)
    yi2 = min(y2, y2p)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)

    union = area1 + area2 - inter

    return inter / union if union != 0 else 0


# -----------------------------
# Sidebar — control panel
# -----------------------------
with st.sidebar:
    st.markdown('<div class="panel-label">Control Panel</div>', unsafe_allow_html=True)
    mode = st.radio("Input source", ["Image", "Video"], label_visibility="visible")

    st.markdown('<div class="panel-label">Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#9A9AA0; line-height:1.6;">
    arch: stenosis_detector_v18<br>
    input: {IMG_SIZE}×{IMG_SIZE}<br>
    loss: huber<br>
    status: <span style="color:#3DDC97;">●</span> loaded
    </div>
    """, unsafe_allow_html=True)

    if mode == "Video":
        st.markdown('<div class="panel-label">Stability Filter</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#9A9AA0; line-height:1.6;">
        iou threshold: {IOU_THRESHOLD}<br>
        stable frames: {STABLE_THRESHOLD}
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div class="scan-header">
    <div class="scan-title">STENOSIS<span>::SCAN</span></div>
    <div class="scan-meta">DETECTION ENGINE v18 — BBOX REGRESSION</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# IMAGE MODE
# -----------------------------
if mode == "Image":
    st.markdown('<div class="section-tag">input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection..."):
            xmin, ymin, xmax, ymax = predict_bbox(image)

        boxed = image.copy()
        cv2.rectangle(boxed, (xmin, ymin), (xmax, ymax), (230, 57, 70), 2)
        boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)

        st.markdown('<div class="section-tag">detection output</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="coord-grid">
            <div class="coord-cell"><div class="coord-label">xmin</div><div class="coord-value">{xmin}</div></div>
            <div class="coord-cell"><div class="coord-label">ymin</div><div class="coord-value">{ymin}</div></div>
            <div class="coord-cell"><div class="coord-label">xmax</div><div class="coord-value">{xmax}</div></div>
            <div class="coord-cell"><div class="coord-label">ymax</div><div class="coord-value">{ymax}</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.image(boxed_rgb, width=600)

# -----------------------------
# VIDEO MODE
# -----------------------------
else:
    st.markdown('<div class="section-tag">input</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded_video is not None:
        st.markdown('<div class="section-tag">processing</div>', unsafe_allow_html=True)

        # Save uploaded video to a temp file so OpenCV can read it
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        in_tmp.write(uploaded_video.read())
        in_tmp.flush()

        cap = cv2.VideoCapture(in_tmp.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 10

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        status_text = st.empty()

        prev_box = None
        stable_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            xmin, ymin, xmax, ymax = predict_bbox(frame)
            current_box = (xmin, ymin, xmax, ymax)

            if prev_box is not None and iou(current_box, prev_box) > IOU_THRESHOLD:
                stable_count += 1
            else:
                stable_count = 0

            prev_box = current_box

            if stable_count >= STABLE_THRESHOLD:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (230, 57, 70), 2)

            out.write(frame)

            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.markdown(
                f'<div class="readout-label">frame {frame_idx} / {total_frames}</div>',
                unsafe_allow_html=True
            )

        cap.release()
        out.release()

        status_text.markdown('<div class="readout-label" style="color:#3DDC97;">complete</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-tag">output</div>', unsafe_allow_html=True)

        with open(out_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            label="Download processed video",
            data=video_bytes,
            file_name="output_video.mp4",
            mime="video/mp4"
        )

        os.unlink(in_tmp.name)
        os.unlink(out_path)
