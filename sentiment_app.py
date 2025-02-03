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
    h1 {
        color: #6a0572;
        text-align: center;
        font-size: 3em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important; /* Stay white always */
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 1.2em;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        color: white !important; /* Stay white on hover */
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
    <h1 style="font-size: 41px; text-align: center;">Apple AirPods Sentiment Analysis</h1>
    """,
    unsafe_allow_html=True
)

image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
]

cols = st.columns(5) # Create 5 columns

for i, url in enumerate(image_urls):
    with cols[i]:
        st.image(url, width=100) # Display each image in its column, adjust width as needed

user_input = st.text_area("Enter your AirPods review here") # Changed text area placeholder

if st.button("🔍 Analyze Sentiment"): # Changed button text and icon
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        sentiment_label = get_sentiment_label(sentiment_probs[0])  # Get the label for the highest probability
        background_color = get_background_color(sentiment_label)  # Get the background color for the sentiment
        st.markdown(
            f"""
            <div style="background-color:{background_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3><span style="font-weight: bold;">Sentiment</span>: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
