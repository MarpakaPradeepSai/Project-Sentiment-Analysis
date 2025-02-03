# sentiment_app.py
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Update the path to point to your ALBERT_Model directory
    model = AutoModelForSequenceClassification.from_pretrained('./ALBERT_Model')
    tokenizer = AutoTokenizer.from_pretrained('./ALBERT_Model')
    return model, tokenizer

model, tokenizer = load_model()

# Set up class labels
class_labels = ['Negative', 'Neutral', 'Positive']

# Streamlit app
st.title("✈️ Airline Sentiment Analysis with ALBERT ✨")
st.write("Analyze customer reviews for airline services")

def predict_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    
    return class_labels[predicted_class], logits

# User input
user_input = st.text_area("Enter your airline review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        # Get prediction
        sentiment, logits = predict_sentiment(user_input)
        
        # Display result with emoji
        emoji = ""
        if sentiment == "Positive":
            emoji = "😊"
        elif sentiment == "Neutral":
            emoji = "😐"
        else:
            emoji = "😞"
            
        st.subheader(f"Predicted Sentiment: {sentiment} {emoji}")
        
        # Add confidence scores (optional)
        with st.expander("See detailed confidence scores"):
            scores = torch.nn.functional.softmax(logits, dim=1)[0]
            for i, score in enumerate(scores):
                st.write(f"{class_labels[i]}: {score:.2%}")
    else:
        st.warning("Please enter a review to analyze!")

# Add some styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)
