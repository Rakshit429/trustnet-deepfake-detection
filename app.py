import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import imageio
from facenet_pytorch import MTCNN
from transformers import ViTImageProcessor, AutoModelForImageClassification

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
    - 🖼️ Frame Sampling  
    - 🙂 Face Detection  
    - 🧠 Deepfake Analysis  
    - 🧾 Verdict  
    """)
    st.info("⚠️ CPU-only (Streamlit Cloud)")
    st.success("✅ Fully Cloud Compatible")

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "uploaded_videos"
FRAMES_FOLDER = "extracted_frames"
FACES_FOLDER = "extracted_faces"

for f in [UPLOAD_FOLDER, FRAMES_FOLDER, FACES_FOLDER]:
    os.makedirs(f, exist_ok=True)

# =========================
# UPLOAD
# =========================
uploaded_video = st.file_uploader(
    "📤 Upload a video",
    type=["mp4", "avi", "mov"]
)

# =========================
# FRAME EXTRACTION
# =========================
def extract_frames(video_path, out_dir, max_cap=200):
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()

    fps = meta.get("fps", 30)
    total = meta.get("nframes", 300)

    desired = min(int((total / fps) * 0.3), max_cap)
    interval = max(total // max(desired, 1), 1)

    count = 0
    for i, frame in enumerate(reader):
        if i % interval == 0:
            Image.fromarray(frame).save(f"{out_dir}/frame_{count}.jpg")
            count += 1
            if count >= desired:
                break

    reader.close()

# =========================
# FACE DETECTOR (SAFE)
# =========================
@st.cache_resource
def load_face_detector():
    return MTCNN(keep_all=True, device="cpu")

def extract_faces(frames_dir, faces_dir):
    detector = load_face_detector()
    face_count = 0

    for img in os.listdir(frames_dir):
        image = Image.open(os.path.join(frames_dir, img)).convert("RGB")
        faces = detector(image)

        if faces is None:
            continue

        for face in faces:
            face = (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(face).save(f"{faces_dir}/face_{face_count}.jpg")
            face_count += 1

            if face_count >= 50:
                return face_count

    return face_count

# =========================
# BLUR SCORE
# =========================
def blur_score(path):
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    gx, gy = np.gradient(img)
    return float(np.mean(gx**2 + gy**2))

# =========================
# DEEPFAKE MODEL
# =========================
@st.cache_resource
def load_model():
    name = "prithivMLmods/Deep-Fake-Detector-Model"
    processor = ViTImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name)
    model.eval()
    return processor, model

def predict_face(path):
    processor, model = load_model()
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]

    return probs[1].item(), probs[0].item()

# =========================
# ANALYSIS
# =========================
def analyze_faces(folder):
    real, fake, blur = [], [], []

    for f in os.listdir(folder)[:50]:
        p = os.path.join(folder, f)
        r, f_ = predict_face(p)
        real.append(r)
        fake.append(f_)
        blur.append(blur_score(p))

    if not real:
        return None

    return np.mean(real), np.mean(fake), np.var(fake), np.mean(blur)

# =========================
# PIPELINE
# =========================
if uploaded_video:
    st.video(uploaded_video)

    video_path = f"{UPLOAD_FOLDER}/{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    extract_frames(video_path, FRAMES_FOLDER)
    faces_count = extract_faces(FRAMES_FOLDER, FACES_FOLDER)

    result = analyze_faces(FACES_FOLDER)

    if result is None:
        st.warning("No faces detected.")
    else:
        real, fake, var, blur = result

        st.metric("🟢 Real", f"{real:.2f}")
        st.metric("🔴 Fake", f"{fake:.2f}")

        if fake > real:
            st.error("🚨 Deepfake Detected")
        else:
            st.success("✅ Appears Authentic")
