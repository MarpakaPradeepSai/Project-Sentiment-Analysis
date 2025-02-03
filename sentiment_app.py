import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests

# Define URLs to fetch the model and tokenizer from your GitHub repository
model_url = "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/raw/main/ALBERT_Model"
tokenizer_url = model_url

# Load model and tokenizer from GitHub repository
def load_model_and_tokenizer(model_url, tokenizer_url):
    # Download model and tokenizer files
    model_files = ["config.json", "model.safetensors", "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"]
    for file in model_files:
        file_url = f"{model_url}/{file}"
        response = requests.get(file_url)
        with open(file, 'wb') as f:
            f.write(response.content)
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('./ALBERT_Model', num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained('./ALBERT_Model')
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_url, tokenizer_url)

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get the model predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Map the prediction to sentiment label
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_map[predictions.item()]
    
    return sentiment

# Streamlit UI
st.title("Sentiment Analysis with ALBERT")
st.write("Enter some text, and I'll predict whether it's positive, neutral, or negative.")

# User input
input_text = st.text_area("Enter Review Text", "Type here...")

# Button to predict sentiment
if st.button("Analyze Sentiment"):
    if input_text:
        # Get sentiment prediction
        sentiment = predict_sentiment(input_text)
        st.write(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text for analysis.")
