import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ----------------------------
# Title & Description
# ----------------------------
st.title("📰 Multilingual Fake News Detection")
st.markdown(
    """
    This system automatically detects the **language of the news** and uses  
    **DistilBERT (English)** or **IndicBERT (Indian Languages)**  
    to classify news as **Real**, **Fake**, or **Suspicious**.
    """
)

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Models (Cached)
# ----------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    indic_path = os.path.join(base_dir, "fake_news_indicbert/fake_news_indicbert")
    en_path = os.path.join(base_dir, "fake_news_distilbert/fake_news_distilbert")

    #INdicBERT
    indic_tokenizer = AutoTokenizer.from_pretrained(indic_path)
    indic_model = AutoModelForSequenceClassification.from_pretrained(indic_path)
    indic_model.to(device).eval()

    #DistilBERT
    en_tokenizer = AutoTokenizer.from_pretrained(en_path)
    en_model = AutoModelForSequenceClassification.from_pretrained(en_path)
    en_model.to(device).eval()

    return indic_tokenizer, indic_model, en_tokenizer, en_model

indic_tokenizer, indic_model, en_tokenizer, en_model = load_models()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_news(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    if lang == "en":
        model = en_model
        tokenizer = en_tokenizer
        model_name = "DistilBERT (English)"
    else:
        model = indic_model
        tokenizer = indic_tokenizer
        model_name = "IndicBERT (Indian Languages)"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    real_p = probs[0][0].item()
    fake_p = probs[0][1].item()

    if fake_p >= 0.7:
        decision = "🟥 FAKE NEWS"
    elif fake_p >= 0.4:
        decision = "🟧 SUSPICIOUS (Needs Fact Check)"
    else:
        decision = "🟩 LIKELY REAL NEWS"

    return lang, model_name, real_p, fake_p, decision

# ----------------------------
# User Input
# ----------------------------
news_text = st.text_area(
    "📝 Enter News Text",
    height=200,
    placeholder="Paste news text here (English or any Indian language)..."
)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Analyze News"):
    if not news_text.strip():
        st.warning("Please enter some news text.")
    else:
        with st.spinner("Analyzing..."):
            lang, model_name, real_p, fake_p, decision = predict_news(news_text)

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Language", lang)
            st.metric("Model Used", model_name)

        with col2:
            st.metric("Real Probability", f"{real_p:.2f}")
            st.metric("Fake Probability", f"{fake_p:.2f}")

        st.markdown("---")
        st.markdown(f"### Final Decision: **{decision}**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Multilingual Fake News Detection using Transformers")
