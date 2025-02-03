import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch

# GitHub repository details (replace with your actual repo and directory)
repo_owner = 'MarpakaPradeepSai'  # Replace with your GitHub username
repo_name = 'Project-Sentiment-Analysis'  # Replace with your GitHub repository name
model_dir = 'ALBERT_Model'  # Directory in your repo where model files are located

# Construct the model path for GitHub
model_path = f"https://huggingface.co/{repo_owner}/{repo_name}/tree/main/{model_dir}"
model_path_raw = f"https://huggingface.co/{repo_owner}/{repo_name}/resolve/main/{model_dir}" # Use resolve to directly access files

@st.cache_resource
def load_model_tokenizer():
    """Loads the fine-tuned ALBERT model and tokenizer from the GitHub repository."""
    tokenizer = AlbertTokenizer.from_pretrained(model_path_raw)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_raw)
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    """Predicts the sentiment of the given text using the loaded model."""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    return sentiment_labels[predicted_class_id]

st.title("Sentiment Analysis of AirPods Reviews")
st.write("Enter your review text below to get the sentiment analysis.")

review_text = st.text_area("Enter Review Text Here")

if st.button("Analyze Sentiment"):
    if review_text:
        with st.spinner("Analyzing sentiment..."):
            sentiment = predict_sentiment(review_text)
        st.write("### Predicted Sentiment:")
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.warning("Please enter some review text to analyze.")
