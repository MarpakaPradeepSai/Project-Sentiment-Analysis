import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch

# Update the MODEL_PATH to point to your local directory
# Define the model path relative to your repo structure
MODEL_PATH = './ALBERT_Model'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)

# Function to perform sentiment analysis
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to predicted label (0: Negative, 1: Neutral, 2: Positive)
    prediction = torch.argmax(logits, dim=-1).item()
    
    # Mapping sentiment back to string
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[prediction]

# Streamlit UI
st.title("Sentiment Analysis with ALBERT")
st.write("This application analyzes the sentiment of your input text using a fine-tuned ALBERT model.")

# User input
user_input = st.text_area("Enter your text here:")

if user_input:
    # Get sentiment prediction
    sentiment = predict_sentiment(user_input)
    
    # Display prediction result
    st.write(f"Predicted Sentiment: {sentiment}")
