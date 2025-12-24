import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


import streamlit as st
import os
import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="TrustNet – Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center;'>🛡️ TrustNet</h1>
    <h4 style='text-align: center; color: gray;'>
    Agentic AI for Deepfake Detection & Authenticity Verification
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ System Overview")
    st.markdown("""
    **Pipeline**
    - 🎥 Video Upload  
    - 🖼️ Adaptive Frame Sampling  
    - 🙂 Face Detection  
    - 🧠 Deepfake Analysis  
    - 🧾 Explainable Verdict  
    """)
    st.info("⚠️ CPU-only inference (Streamlit Cloud)")
    st.success("✅ Explainable Analysis Active")

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "uploaded_videos"
FRAMES_FOLDER = "extracted_frames"
FACES_FOLDER = "extracted_faces"

for folder in [UPLOAD_FOLDER, FRAMES_FOLDER, FACES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# =========================
# UPLOAD
# =========================
st.subheader("📤 Upload Media for Verification")
uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov"]
)

# =========================
# FRAME EXTRACTION
# =========================
def extract_frames(video_path, output_folder, max_cap=200):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)

    duration_sec = total_frames / fps
    desired_frames = min(int(duration_sec * 0.3), max_cap)
    interval = max(total_frames // max(desired_frames, 1), 1)

    saved, count = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(
                os.path.join(output_folder, f"frame_{saved}.jpg"),
                frame
            )
            saved += 1
            if saved >= desired_frames:
                break
        count += 1

    cap.release()
    return saved

# =========================
# FACE EXTRACTION
# =========================
@st.cache_resource
def load_face_detector():
    return MTCNN()

def extract_faces(frames_folder, faces_folder):
    detector = load_face_detector()
    os.makedirs(faces_folder, exist_ok=True)
    face_count = 0

    for img_name in os.listdir(frames_folder):
        img_path = os.path.join(frames_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        faces = detector.detect_faces(image)
        for face in faces:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)
            face_img = image[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            cv2.imwrite(
                os.path.join(faces_folder, f"face_{face_count}.jpg"),
                face_img
            )
            face_count += 1

            if face_count >= 50:  # HARD LIMIT FOR CLOUD
                return face_count

    return face_count

# =========================
# BLUR SCORE
# =========================
def blur_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

# =========================
# DEEPFAKE MODEL
# =========================
@st.cache_resource
def load_deepfake_model():
    model_name = "prithivMLmods/Deep-Fake-Detector-Model"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model

def predict_face(image_path):
    processor, model = load_deepfake_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    return probs[1].item(), probs[0].item()

# =========================
# ANALYSIS
# =========================
def analyze_faces(faces_folder):
    real_scores, fake_scores, blur_scores = [], [], []

    face_files = os.listdir(faces_folder)[:50]
    if not face_files:
        return None, None, None, None

    for face in face_files:
        path = os.path.join(faces_folder, face)
        real, fake = predict_face(path)
        real_scores.append(real)
        fake_scores.append(fake)
        blur_scores.append(blur_score(path))

    return (
        float(np.mean(real_scores)),
        float(np.mean(fake_scores)),
        float(np.var(fake_scores)),
        float(np.mean(blur_scores))
    )

# =========================
# PIPELINE
# =========================
if uploaded_video:
    st.markdown("### 🔄 Processing Pipeline")
    progress = st.progress(0)

    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video(video_path)
    progress.progress(20)

    video_name = os.path.splitext(uploaded_video.name)[0]
    frames_output = os.path.join(FRAMES_FOLDER, video_name)
    faces_output = os.path.join(FACES_FOLDER, video_name)

    extract_frames(video_path, frames_output)
    progress.progress(40)

    faces_count = extract_faces(frames_output, faces_output)
    progress.progress(60)

    st.subheader("🧠 Deepfake Analysis")
    real_score, fake_score, variance, avg_blur = analyze_faces(faces_output)
    progress.progress(100)

    if real_score is None:
        st.warning("No faces detected. Try a clearer video.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Real Probability", f"{real_score:.2f}")
        col2.metric("🔴 Fake Probability", f"{fake_score:.2f}")
        col3.metric("📊 Consistency", f"{variance:.4f}")

        st.markdown("### 🧾 Final Verdict")
        margin = abs(fake_score - real_score)

        if fake_score > real_score and margin > 0.1:
            st.error("🚨 High Confidence Deepfake Detected")
        elif real_score > fake_score and margin > 0.1:
            st.suc
