import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model and tokenizer
try:
    model_dir = './ALBERT_Model'  # Assuming the ALBERT_Model directory is in the same directory as your streamlit script
    tokenizer = AlbertTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    st.success("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Sentiment mapping dictionary
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    """
    Predicts the sentiment of the given text using the loaded ALBERT model.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        str: The predicted sentiment label (e.g., "Positive", "Negative", "Neutral").
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return sentiment_labels[predicted_class_id]

# Streamlit app
st.title("Sentiment Analysis with Fine-tuned ALBERT Model")

user_input = st.text_area("Enter your review text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            predicted_sentiment = predict_sentiment(user_input)
        st.write("### Predicted Sentiment:")
        if predicted_sentiment == "Positive":
            st.success(f"Positive 😊")
        elif predicted_sentiment == "Negative":
            st.error(f"Negative 😠")
        else:
            st.info(f"Neutral 😐")
    else:
        st.warning("Please enter some text to analyze.")
