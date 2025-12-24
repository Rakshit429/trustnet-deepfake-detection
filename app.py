import streamlit as st
import os
import shutil
import imageio
import math
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "uploaded_videos"
FRAMES_FOLDER = "frames"
FACES_FOLDER = "faces"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,
    device=device
)

# =========================
# CLEANUP
# =========================
def clean_dirs():
    for d in [FRAMES_FOLDER, FACES_FOLDER]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

# =========================
# FRAME EXTRACTION (FIXED)
# =========================
def extract_frames(video_path, out_dir, max_frames=200):
    os.makedirs(out_dir, exist_ok=True)

    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
    except Exception as e:
        st.error(f"Video decode failed: {e}")
        return 0

    count = 0
    stride = 1

    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps", 30)

        if fps and isinstance(fps, (int, float)) and not math.isinf(fps):
            stride = max(int(fps // 3), 1)
    except Exception:
        stride = 1

    for i, frame in enumerate(reader):
        if i % stride == 0:
            Image.fromarray(frame).save(f"{out_dir}/frame_{count}.jpg")
            count += 1
            if count >= max_frames:
                break

    reader.close()
    return count

# =========================
# FACE EXTRACTION
# =========================
def extract_faces(frames_dir, faces_dir):
    os.makedirs(faces_dir, exist_ok=True)
    face_count = 0

    for img_name in sorted(os.listdir(frames_dir)):
        img_path = os.path.join(frames_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        faces = mtcnn(img)

        if faces is not None:
            for face in faces:
                face_img = face.permute(1, 2, 0).int().numpy()
                Image.fromarray(face_img.astype("uint8")).save(
                    f"{faces_dir}/face_{face_count}.jpg"
                )
                face_count += 1

    return face_count

# =========================
# FAKE ANALYSIS (DEMO LOGIC)
# =========================
def analyze_faces(faces_dir):
    total = len(os.listdir(faces_dir))
    if total == 0:
        return "No faces detected"

    # Placeholder logic
    fake_ratio = np.random.uniform(0.3, 0.7)

    if fake_ratio > 0.5:
        return f"⚠️ Likely Deepfake ({fake_ratio:.2%})"
    else:
        return f"✅ Likely Real ({1 - fake_ratio:.2%})"

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="TrustNet Deepfake Detector", layout="centered")
st.title("🎭 TrustNet – Deepfake Video Detector")

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"]
)

if uploaded_video:
    clean_dirs()

    video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.video(video_path)

    with st.spinner("Extracting frames..."):
        frames = extract_frames(video_path, FRAMES_FOLDER)

    st.success(f"Extracted {frames} frames")

    with st.spinner("Detecting faces..."):
        faces = extract_faces(FRAMES_FOLDER, FACES_FOLDER)

    st.success(f"Detected {faces} faces")

    with st.spinner("Analyzing authenticity..."):
        result = analyze_faces(FACES_FOLDER)

    st.subheader("Result")
    st.write(result)
