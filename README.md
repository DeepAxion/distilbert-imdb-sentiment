# 🧠 Sentiment Analizer — Fine-tuned Transformer for NLP

A high-performance sentiment analysis model fine-tuned on the IMBD Movie Review Dataset using `DistilBERT`. Built for fast, reliable predictions and deployed with an interactive **Streamlit** demo.

[![🤗 Model on Hugging Face](https://img.shields.io/badge/View%20on-Hugging%20Face-blue)](https://huggingface.co/your-username/your-model-name)
[![🚀 Live Demo](https://img.shields.io/badge/Try%20Live-Streamlit%20App-success)](https://your-streamlit-app-link.com)

---

## 📖 Overview

This project fine-tunes a pretrained transformer (DistilBERT) on IMBD Movie Review Dataset to solve binary sentiment classification. It compares traditional NLP techniques (e.g., TF-IDF + logistic regression) with the transformer-based model achieving state-of-the-art performance.

---

## 🔍 Features

- ✅ Preprocessing with `spaCy`: tokenization, lemmatization, stopword removal  
- 📊 Baseline using TF-IDF and logistic regression  
- 🚀 Fine-tuning DistilBERT using Hugging Face Transformers  
- 🖥️ Live demo with `streamlit` for real-time predictions  

---

## 📊 Performance

| Model                    | Accuracy|Precision|Recall|F1|
|--------------------------|---------|---------|------|-----|
| TF-IDF + Logistic Regression | 89%|89%|89%|89%|
| Fine-tuned DistilBERT        | 94%|94%|94%|93%|

---

## 🚀 Try the Model

- 🤗 [Hugging Face Model](https://huggingface.co/DeepAxion/distilbert-imdb-sentiment)  
- 🌐 [Streamlit Demo](https://distilbert-imdb-sa.streamlit.app/)

---

## 🛠️ Usage

### Load from Hugging Face
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "DeepAxion/distilbert-imdb-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("Your input text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

### Run streamlit app locally

1. Clone the repository
```
git clone https://github.com/DeepAxion/distilbert-imdb-sentiment.git
```
2. Go to the project directory
```
cd distilbert-imdb-sentiment
```

3. Install requirements
```
pip install -r requirements.txt
```
4. Run streamlit app
```
streamlit run app.py
```

Demo:

![streamlit_app.png](streamlit_app.png)
## 🧰 Tech Stack

- `spaCy` – Text preprocessing
- `scikit-learn` – Baseline model training
- `Hugging Face Transformers` – Fine-tuning and deployment
- `Streamlit` – Web interface for demo

## 📬 Contact
For feedback or collaboration, reach out:

- LinkedIn: [Anthony Nguyen](https://www.linkedin.com/in/nbk2003/)

- Hugging Face: [DeepAxion](https://huggingface.co/DeepAxion)


## 📄 License
This project is licensed under the  `MIT licensed`. See `LICENSE` for more details.