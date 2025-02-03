import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch
import requests

# Define a function to load the model and tokenizer from GitHub
def load_model_from_github():
    model_url = "https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/tree/main/ALBERT_Model"
    
    # Load model and tokenizer from the GitHub repository
    model = AutoModelForSequenceClassification.from_pretrained(model_url)
    tokenizer = AlbertTokenizer.from_pretrained(model_url)
    return model, tokenizer

# Load the pre-trained model and tokenizer
model, tokenizer = load_model_from_github()

# Function to perform sentiment analysis
def predict_sentiment(review_text):
    # Tokenize the input text
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Map predicted class id to sentiment label
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class_id]

# Set up Streamlit app interface
st.title("Sentiment Analysis with ALBERT Model")
st.write("Enter a review to get its sentiment (Negative, Neutral, or Positive):")

# Text input field for user to input review
user_input = st.text_area("Review Text", height=150)

if st.button('Analyze Sentiment'):
    if user_input:
        # Get sentiment prediction
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review text to analyze.")
