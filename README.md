# AI Phishing Email Detector

An AI-powered email classification web app built with **Python**, **scikit-learn**, and **Streamlit**.

This project uses a trained machine learning model to analyse email subject lines and body text, then predict:

- whether the email is **Phishing** or **Normal**
- whether the email appears **AI-assisted / AI-generated** or **Human-written**

---

## Project Structure

```text
Dissertation_AI_Cyber/
  webapp/
    app.py
    requirements.txt
    README.md
    models/
      phishing_ai_detector.pkl
    .venv/



python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit scikit-learn joblib pandas numpy
python -m streamlit run app.py
