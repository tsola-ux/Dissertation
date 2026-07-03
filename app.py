import os
import joblib
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Cyber Defence Tool", page_icon="🛡️", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_ai_detector.pkl")
BILSTM_MODEL_PATH = os.path.join(BASE_DIR, "models", "bilstm_phishing_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "bilstm_tokenizer.pkl")
DEEPFAKE_MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_xception.keras")

MAX_LEN = 512
IMG_SIZE = 299

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

@st.cache_resource
def load_lr_model():
    return joblib.load(LR_MODEL_PATH)

@st.cache_resource
def load_bilstm_model():
    model = tf.keras.models.load_model(BILSTM_MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

@st.cache_resource
def load_deepfake_model():
    return tf.keras.models.load_model(DEEPFAKE_MODEL_PATH)

@st.cache_resource
def load_face_detector():
    import cv2
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def combined_label(is_phishing, is_ai):
    if is_phishing == 0 and is_ai == 0:
        return "Normal + Human-written"
    elif is_phishing == 0 and is_ai == 1:
        return "Normal + AI-assisted"
    elif is_phishing == 1 and is_ai == 0:
        return "Phishing + Human-written"
    else:
        return "Phishing + AI-assisted"

def predict_deepfake(image_bytes, model):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    img = load_img(tmp_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array, verbose=0)[0][0]
    os.unlink(tmp_path)
    label = "Real" if prob >= 0.5 else "Fake (Deepfake)"
    confidence = prob if prob >= 0.5 else (1 - prob)
    return label, float(confidence), float(prob)

def detect_and_classify_face(frame_rgb, face_detector, deepfake_model):
    """Detect face in frame, crop it, classify as real/fake. Falls back to full frame if no face found."""
    import cv2
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

    if len(faces) > 0:
        # Use the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.2 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_rgb.shape[1], x + w + pad)
        y2 = min(frame_rgb.shape[0], y + h + pad)
        region = frame_rgb[y1:y2, x1:x2]
    else:
        # No face found — use full frame as fallback
        region = frame_rgb

    resized = cv2.resize(region, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = deepfake_model.predict(arr, verbose=0)[0][0]
    return {"prob": float(prob), "label": "Real" if prob >= 0.5 else "Fake"}


# ── App ──
st.title("🛡️ AI Cyber Defence Tool")
st.write("Detect phishing emails, AI-generated content, and deepfake images/videos.")

tab1, tab2 = st.tabs(["📧 Email Phishing Detector", "🎭 Deepfake Detector"])

# ── TAB 1: EMAIL ──
with tab1:
    st.header("📧 Phishing & AI-Assisted Email Detector")

    if TF_AVAILABLE:
        email_model_options = ["Logistic Regression (Baseline)", "BiLSTM (Advanced)"]
    else:
        email_model_options = ["Logistic Regression (Baseline)"]

    email_model_choice = st.selectbox("Select Model", email_model_options, key="email_model")

    if not TF_AVAILABLE and len(email_model_options) == 1:
        st.info("BiLSTM requires TensorFlow. Logistic Regression is active.")

    subject = st.text_input("Email Subject", key="email_subject")
    body = st.text_area("Email Body", height=200, key="email_body")

    if st.button("Analyze Email", key="email_btn"):
        if not subject.strip() and not body.strip():
            st.warning("Please enter an email subject or body.")
        else:
            text = f"Subject: {subject}\n\nBody: {body}"

            if email_model_choice == "Logistic Regression (Baseline)":
                model = load_lr_model()
                pred = model.predict([text])[0]
                phishing_pred = int(pred[0])
                ai_pred = int(pred[1])
                try:
                    probas = model.predict_proba([text])
                    phishing_conf = probas[0][0][1]
                    ai_conf = probas[1][0][1]
                except Exception:
                    phishing_conf = None
                    ai_conf = None
            else:
                bilstm_model, tokenizer = load_bilstm_model()
                seq = pad_sequences(tokenizer.texts_to_sequences([text]),
                    maxlen=MAX_LEN, padding="post", truncating="post")
                phish_prob, ai_prob = bilstm_model.predict(seq, verbose=0)
                phishing_conf = float(phish_prob[0][0])
                ai_conf = float(ai_prob[0][0])
                phishing_pred = int(phishing_conf >= 0.5)
                ai_pred = int(ai_conf >= 0.5)

            phishing_result = "Phishing" if phishing_pred == 1 else "Normal"
            ai_result = "AI-assisted / AI-generated" if ai_pred == 1 else "Human-written"

            st.subheader("Prediction Result")
            st.write(f"**Model Used:** {email_model_choice}")
            st.write(f"**Email Type:** {phishing_result}")
            st.write(f"**Writing Style:** {ai_result}")
            st.write(f"**Combined Class:** {combined_label(phishing_pred, ai_pred)}")
            if phishing_conf is not None and ai_conf is not None:
                st.write(f"**Phishing Confidence:** {phishing_conf:.2%}")
                st.write(f"**AI-assisted Confidence:** {ai_conf:.2%}")

# ── TAB 2: DEEPFAKE ──
with tab2:
    st.header("🎭 Deepfake Face Detector")
    st.write("Upload an image or video to check if faces are real or AI-generated.")

    if not TF_AVAILABLE:
        st.error("Deepfake detection requires TensorFlow, which is not available on this system.")
    else:
        deepfake_model = load_deepfake_model()
        face_detector = load_face_detector()

        upload_type = st.radio("Upload type", ["Image", "Video"], horizontal=True, key="upload_type")

        if upload_type == "Image":
            uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"], key="img_upload")

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", width=300)

                if st.button("Analyze Image", key="img_btn"):
                    with st.spinner("Analyzing..."):
                        label, confidence, raw_prob = predict_deepfake(uploaded_file.getvalue(), deepfake_model)

                    st.subheader("Prediction Result")
                    if "Fake" in label:
                        st.error(f"🚨 **{label}** — Confidence: {confidence:.2%}")
                    else:
                        st.success(f"✅ **{label}** — Confidence: {confidence:.2%}")
                    st.progress(raw_prob)
                    st.caption(f"Score: {raw_prob:.4f} (>0.5 = Real, <0.5 = Fake)")

        else:  # Video
            uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload")

            if uploaded_video is not None:
                st.video(uploaded_video)
                frame_interval = st.slider("Analyze every N-th frame", 1, 60, 1, key="frame_int")

                if st.button("Analyze Video", key="vid_btn"):
                    import tempfile
                    import cv2

                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(uploaded_video.getvalue())
                        video_path = tmp.name

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if total_frames > 0 and fps > 0:
                        st.write(f"Video: {total_frames} frames, {fps:.0f} FPS, {total_frames/fps:.1f}s")

                    results = []
                    frame_count = 0
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_count % frame_interval == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result = detect_and_classify_face(frame_rgb, face_detector, deepfake_model)
                            results.append(result)

                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                            status_text.text(f"Analyzed {len(results)} frames...")

                        frame_count += 1

                    cap.release()
                    os.unlink(video_path)
                    progress_bar.progress(1.0)

                    if results:
                        fake_count = sum(1 for r in results if r["label"] == "Fake")
                        real_count = len(results) - fake_count
                        fake_pct = fake_count / len(results) * 100
                        avg_prob = np.mean([r["prob"] for r in results])

                        st.subheader("Video Analysis Result")

                        if fake_pct >= 60:
                            st.error(f"## 🚨 Likely Deepfake\n**{fake_pct:.1f}% of frames flagged as fake**")
                        elif fake_pct >= 30:
                            st.warning(f"## ⚠️ Uncertain\n**{fake_pct:.1f}% of frames flagged as fake**")
                        else:
                            st.success(f"## ✅ Likely Authentic\n**Only {fake_pct:.1f}% of frames flagged**")

                    else:
                        st.warning("No faces detected in the video frames. Try a video with clear, front-facing faces.")


