import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch
import requests
import os

# Function to download model files from GitHub
def download_file_from_github(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"Failed to download {url}: Status code {response.status_code}")

# URLs of your ALBERT model files on GitHub
repo_url = 'https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/raw/main/ALBERT_Model'  # Updated repo URL for ALBERT model
files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json'] # Files in ALBERT_Model directory

# Create model directory if it doesn't exist
model_dir = './albert_model' # Changed model directory name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Download each file
for file in files:
    download_file_from_github(f"{repo_url}/{file}", os.path.join(model_dir, file))

# Load tokenizer and model
try:
    tokenizer = AlbertTokenizer.from_pretrained(model_dir) # Changed to AlbertTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3) # Changed to AutoModelForSequenceClassification and added num_labels
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1) # Keep dim=1 for binary/multi-class
    return probs.detach().numpy()

# Function to map probabilities to sentiment labels and emojis
def get_sentiment_label(probs):
    sentiment_mapping = ["Negative 😡", "Neutral 😐", "Positive 😊"] # Assuming 3 labels: Negative, Neutral, Positive
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# Function to get background color based on sentiment
def get_background_color(label):
    if "Positive" in label:
        return "#C3E6CB"  # Softer green
    elif "Neutral" in label:
        return "#FFE8A1"  # Softer yellow
    else:
        return "#F5C6CB"  # Softer red

# Streamlit app
st.set_page_config(
    page_title="Sentiment Analysis of AirPods Reviews", # Changed title to be more specific
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s, transform 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        transform: scale(1.05);
    }
    .prediction-box {
        border-radius: 25px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
    }
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="font-size: 41px; text-align: center;">Sentiment Analysis of AirPods Reviews</h1>  # Changed title to be more specific
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <img src="https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF" alt="Apple AirPods" class="center-image" width="400">
    """,
    unsafe_allow_html=True
)

user_input = st.text_area("Enter your AirPods review here") # Changed text area placeholder

if st.button("Analyze"):
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        sentiment_label = get_sentiment_label(sentiment_probs[0])  # Get the label for the highest probability
        background_color = get_background_color(sentiment_label)  # Get the background color for the sentiment
        st.markdown(
            f"""
            <div style="background-color:{background_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3>Sentiment: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
