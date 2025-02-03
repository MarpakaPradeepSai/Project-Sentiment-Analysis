import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer from local directory
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained('./ALBERT_Model')
    tokenizer = AlbertTokenizer.from_pretrained('./ALBERT_Model')
    return model, tokenizer

model, tokenizer = load_model()

# Sentiment labels
class_labels = ['Negative', 'Neutral', 'Positive']

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probs).item()
    return class_labels[pred_class], probs[0].tolist()

# Streamlit UI
st.title("✈️ Airline Sentiment Analysis with ALBERT")
st.write("Analyze passenger reviews for Negative, Neutral, or Positive sentiment")

review = st.text_area("Enter your airline review here:", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze")
    else:
        prediction, probabilities = predict_sentiment(review)
        emoji = "😞" if prediction == "Negative" else "😐" if prediction == "Neutral" else "😊"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Prediction:")
            st.markdown(f"<h2 style='color: {'red' if prediction == 'Negative' else 'orange' if prediction == 'Neutral' else 'green'};'>{emoji} {prediction}</h2>", 
                        unsafe_allow_html=True)
        
        with col2:
            st.subheader("Confidence:")
            st.write(f"Negative: {probabilities[0]*100:.1f}%")
            st.write(f"Neutral: {probabilities[1]*100:.1f}%")
            st.write(f"Positive: {probabilities[2]*100:.1f}%")
