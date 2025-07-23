import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load tokenizer and model once
@st.cache_resource # cache the data
def load_model():
    # define the base directory where your app.py is located
    base_dir = os.path.dirname(__file__)

    # define the paths to your local model and tokenizer folders relative to base_dir
    model_path = os.path.join(base_dir, "distilbert_finetuned_imdb")
    tokenizer_path = os.path.join(base_dir, "distilbert_finetuned_tokenizer_imdb")
    
    # load the tokenizer from its correct folder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # load the model from its correct folder
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # put the model in eval mode
    model.eval()
    
    return tokenizer, model

tokenizer, model = load_model()

# UI Design
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #6c63ff;'>🎭 Your Mood Analyzer 🎭 </h1>
    <p style='text-align: center;'>Get real-time sentiment and confidence scores for your text!</p>
""", unsafe_allow_html=True)


# input box label with a custom class
st.markdown(
    '<p class="input-label-gap" style="font-size: 24px; font-weight: bold;">👨‍🍼 Let\' see how yah feeling:</p>',
    unsafe_allow_html=True
)

# get the text
text = st.text_area("", height=200) # Empty string as label to avoid duplication

# custom CSS to enlarge text inside the text area
st.markdown("""
    <style>
    textarea {
        font-size: 20px !important;
        line-height: 1.6;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Target the paragraph tag used for the label */
    .input-label-gap {
        margin-top: 30px !important; /* Specific gap below this label */
        margin-bottom: -30px
    }
    
    .stButton > button > div > p {
        font-size: 18px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# analyze button
if st.button("🔍 Analyze", use_container_width=True):
    if text.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

            # sentiment Label
            sentiment = "Positive 😊" if pred == 1 else "Negative 😠"
            color = "#66CDAA" if pred == 1 else "#ff6b6b"

            st.markdown(f"<h2 style='color:{color}; text-align:center;'>Your mood is: {sentiment}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Confidence: <strong>{confidence:.2f}</strong></p>", unsafe_allow_html=True)

            # plot confidence bar
            fig, ax = plt.subplots()
            ax.bar(["Negative", "Positive"], probs.numpy(), color=["#ff6b6b", "#66CDAA"])
            ax.set_ylabel("Confidence")
            ax.set_ylim([0, 1])
            st.pyplot(fig)
            
# --- disclaimer ---
st.markdown("<hr>", unsafe_allow_html=True) # horizontal line
st.markdown("<p style='text-align: center; font-style: italic; color: gray;'>Powered by a Fine-tuned DistilBERT Model trained on IMDB Moview Review Dataset.</p>", unsafe_allow_html=True)