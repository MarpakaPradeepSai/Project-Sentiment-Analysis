# app.py
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model_path = './fine_tuned_ALBERT-base-v2_model'  # Path to your saved model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Sentiment class labels
class_labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    return class_labels[predictions[0]]

st.title("Sentiment Analysis with ALBERT")

review_text = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review_text:
        predicted_sentiment = analyze_sentiment(review_text)
        st.write("### Predicted Sentiment:")
        if predicted_sentiment == 'Positive':
            st.success(f"Positive 😊")
        elif predicted_sentiment == 'Negative':
            st.error(f"Negative 😠")
        else:
            st.warning(f"Neutral 😐")
    else:
        st.warning("Please enter a review to analyze.")
