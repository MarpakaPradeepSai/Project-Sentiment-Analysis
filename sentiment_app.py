import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer from your GitHub repository
model_url = "https://huggingface.co/your_github_repo/ALBERT_Model"  # Replace with the actual GitHub model URL
tokenizer_url = "https://huggingface.co/your_github_repo/ALBERT_Model"  # Same here for the tokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_url, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)

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

