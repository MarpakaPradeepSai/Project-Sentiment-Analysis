import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.class_labels = ['Negative', 'Neutral', 'Positive']
        self.model.eval()

    def predict_sentiment(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            predicted_class = self.class_labels[prediction.item()]

            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probabilities[0][prediction].item() * 100

        return predicted_class, confidence

def main():
    st.set_page_config(
        page_title="Review Sentiment Analyzer",
        page_icon="🎭",
        layout="wide"
    )

    st.title("Review Sentiment Analyzer")
    st.write("Using ALBERT model fine-tuned on product reviews")

    # Initialize the sentiment analyzer
    try:
        analyzer = SentimentAnalyzer('./fine_tuned_ALBERT-base-v2_model')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure the model files are in the correct directory")
        return

    # Create input area for the review
    review_text = st.text_area(
        "Enter your review text here:",
        height=150,
        placeholder="Type or paste your review here..."
    )

    # Add analyze button
    if st.button("Analyze Sentiment"):
        if review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    sentiment, confidence = analyzer.predict_sentiment(review_text)
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    # Display the sentiment with corresponding color
                    with col1:
                        sentiment_color = {
                            'Positive': 'green',
                            'Neutral': 'orange',
                            'Negative': 'red'
                        }
                        st.markdown(f"### Sentiment: <span style='color: {sentiment_color[sentiment]}'>{sentiment}</span>", 
                                  unsafe_allow_html=True)
                    
                    # Display the confidence
                    with col2:
                        st.markdown(f"### Confidence: {confidence:.2f}%")

                    # Add a visual separator
                    st.markdown("---")

                    # Display some example interpretations
                    st.subheader("What this means:")
                    interpretations = {
                        'Positive': "The review expresses satisfaction and approval.",
                        'Neutral': "The review expresses a balanced or mixed opinion.",
                        'Negative': "The review expresses dissatisfaction or criticism."
                    }
                    st.write(interpretations[sentiment])

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter some text to analyze")

    # Add a sample reviews section
    with st.expander("Sample Reviews"):
        st.write("""
        Try these sample reviews:
        
        **Positive**: "I absolutely love these! The sound quality is incredible, and they're worth every penny!"
        
        **Neutral**: "They're okay for casual listening, but I've had some minor issues with connectivity."
        
        **Negative**: "Very disappointed with this purchase. The quality doesn't match the price tag at all."
        """)

    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This sentiment analyzer uses a fine-tuned ALBERT-base-v2 model trained on product reviews. 
        The model classifies reviews into three categories:
        - 🟢 Positive
        - 🟡 Neutral
        - 🔴 Negative
        
        The confidence score indicates how certain the model is about its prediction.
        """)

if __name__ == "__main__":
    main()
