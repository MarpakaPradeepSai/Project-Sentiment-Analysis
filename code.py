import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Function to load the model and tokenizer
@st.cache_resource
def load_model_tokenizer():
    model_path = './ALBERT_model'  # Path to your saved model files
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Function to predict sentiment
def predict_sentiment(review_text, model, tokenizer):
    inputs = tokenizer(review_text, padding=True, truncation=True, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = inputs.to(device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    class_labels = ['Negative', 'Neutral', 'Positive']
    return class_labels[predictions[0]]

# Load the model and tokenizer
model, tokenizer = load_model_tokenizer()

# Streamlit app
st.title('Sentiment Analysis with ALBERT')

review_text = st.text_area("Enter your review here:")

if st.button('Analyze Sentiment'):
    if review_text:
        predicted_sentiment = predict_sentiment(review_text, model, tokenizer)
        st.write("### Prediction:")
        if predicted_sentiment == 'Positive':
            st.success(f"Sentiment: {predicted_sentiment} 😃")
        elif predicted_sentiment == 'Negative':
            st.error(f"Sentiment: {predicted_sentiment} 😞")
        else:
            st.info(f"Sentiment: {predicted_sentiment} 😐")
    else:
        st.warning("Please enter a review to analyze.")
