import streamlit as st
import torch
import os
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define URLs to fetch the model and tokenizer files from GitHub
base_url = "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/raw/main/ALBERT_Model"

model_files = [
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json"
]

# Define directory where model will be saved
model_dir = "ALBERT_Model"

# Download model files from GitHub and save to the model directory
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for file in model_files:
    file_url = f"{base_url}/{file}"
    file_path = os.path.join(model_dir, file)
    
    # If file does not exist, download it
    if not os.path.exists(file_path):
        print(f"Downloading {file}...")
        response = requests.get(file_url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"{file} downloaded successfully.")

# Load the model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
