import os
import joblib
import streamlit as st

st.set_page_config(page_title="AI Phishing Email Detector", page_icon="📧", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_ai_detector.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📧 AI Phishing Email Detector")
st.write("Paste an email subject and body to check whether it is phishing or normal, and whether it appears AI-assisted or human-written.")

subject = st.text_input("Email Subject")
body = st.text_area("Email Body", height=250)

def combined_label(is_phishing, is_ai):
    if is_phishing == 0 and is_ai == 0:
        return "Normal + Human-written"
    elif is_phishing == 0 and is_ai == 1:
        return "Normal + AI-assisted"
    elif is_phishing == 1 and is_ai == 0:
        return "Phishing + Human-written"
    else:
        return "Phishing + AI-assisted"

if st.button("Analyze Email"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter an email subject or body.")
    else:
        text = f"Subject: {subject}\n\nBody: {body}"
        pred = model.predict([text])[0]

        phishing_pred = int(pred[0])
        ai_pred = int(pred[1])

        phishing_result = "Phishing" if phishing_pred == 1 else "Normal"
        ai_result = "AI-assisted / AI-generated" if ai_pred == 1 else "Human-written"

        st.subheader("Prediction Result")
        st.write(f"**Email Type:** {phishing_result}")
        st.write(f"**Writing Style:** {ai_result}")
        st.write(f"**Combined Class:** {combined_label(phishing_pred, ai_pred)}")

        # Optional confidence scores
        try:
            probas = model.predict_proba([text])
            phishing_conf = probas[0][0][1]
            ai_conf = probas[1][0][1]

            st.write(f"**Phishing Confidence:** {phishing_conf:.2%}")
            st.write(f"**AI-assisted Confidence:** {ai_conf:.2%}")
        except Exception:
            pass