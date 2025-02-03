import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch
import requests
import os
import time  # For adding a loading spinner

# --- Function to download model files from GitHub ---
def download_file_from_github(url, local_path):
    response = requests.get(url, stream=True) # Use stream=True for potentially large files
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): # Iterate over chunks
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    else:
        st.error(f"Failed to download {url}: Status code {response.status_code}")

# --- URLs of your ALBERT model files on GitHub ---
repo_url = 'https://github.com/MarpakaPradeepSai/Project-Sentiment-Analysis/raw/main/ALBERT_Model'
files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json']

# --- Create model directory if it doesn't exist ---
model_dir = './albert_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# --- Download each file if model directory is empty or files are missing ---
if not os.listdir(model_dir) or any(not os.path.exists(os.path.join(model_dir, file)) for file in files):
    with st.spinner('Downloading and loading model files...'): # Show spinner while downloading
        for file in files:
            download_file_from_github(f"{repo_url}/{file}", os.path.join(model_dir, file))

        # Load tokenizer and model after download is complete
        try:
            tokenizer = AlbertTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
            st.success("Model files downloaded and loaded successfully!") # Success message after loading
        except Exception as e:
            st.error(f"Error loading model after download: {e}")
else:
    try:
        tokenizer = AlbertTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- Function to predict sentiment ---
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

# --- Function to map probabilities to sentiment labels and emojis ---
def get_sentiment_label(probs):
    sentiment_mapping = ["Negative 😡", "Neutral 😐", "Positive 😊"] # Original emojis
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# --- Function to get background color class based on sentiment ---
def get_background_color_class(label):
    if "Positive" in label:
        return "positive"
    elif "Neutral" in label:
        return "neutral"
    else:
        return "negative"

# --- Streamlit app ---
st.set_page_config(
    page_title="AirPods Review Sentiment Analyzer",
    page_icon=":headphones:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for a more attractive look ---
st.markdown(
    """
    <style>
    /* --- General Styles --- */
    body {
        background-color: #f0f2f5; /* Warmer off-white background */
        font-family: 'Open Sans', sans-serif;
        color: #333;
    }
    h1 {
        font-family: 'Nunito', sans-serif;
        color: #4c6ef5; /* Soft blue title color */
        text-align: center;
        font-size: 2.7em;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    h3 {
        font-family: 'Nunito', sans-serif;
        color: #333;
    }
    .stButton>button {
        background: #66a3ff; /* Soft blue button background */
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 1.1em;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    }
    .stButton>button:hover {
        background-color: #4d88e6; /* Darker blue on hover */
        transform: scale(1.03);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        color: white !important;
    }
    .prediction-box {
        border-radius: 25px;
        padding: 15px;
        text-align: center;
        font-size: 18px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .prediction-box h3 {
        margin-bottom: 0; /* Remove default margin for h3 inside prediction box */
    }
    .stTextArea textarea {
        border-radius: 15px;
        border: 1px solid #ced4da;
        padding: 12px;
        background-color: #ffffff;
        box-shadow: inset 2px 2px 5px #e0e0e0; /* Inset shadow for text area */
    }
    .stTextArea textarea::placeholder {
        color: #999;
        font-style: italic;
    }
    .positive {
        background-color: #d4edda; /* Softer green for positive */
        color: #155724;
    }
    .neutral {
        background-color: #fff3cd; /* Softer yellow for neutral */
        color: #85640a;
    }
    .negative {
        background-color: #f8d7da; /* Softer red for negative */
        color: #721c24;
    }
    .airpods-image-row {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .airpods-image-row img {
        width: 100px;
        margin: 0 10px;
        border-radius: 8px; /* Rounded corners for images */
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    """,
    unsafe_allow_html=True
)

# --- App Title ---
st.markdown(
    """
    <h1 style="font-size: 40px; text-align: center;">Apple AirPods Sentiment Analysis</h1>
    """,
    unsafe_allow_html=True
)

# --- AirPods Image Row ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
]

st.markdown('<div class="airpods-image-row">', unsafe_allow_html=True)
for url in image_urls:
    st.image(url, width=100)
st.markdown('</div>', unsafe_allow_html=True)


# --- User Input Text Area ---
user_input = st.text_area("Share your experience with AirPods:")

# --- Analyze Sentiment Button ---
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            time.sleep(0.5)
            sentiment_probs = predict_sentiment(user_input)
            sentiment_label = get_sentiment_label(sentiment_probs[0])
            sentiment_class = get_background_color_class(sentiment_label)

        st.divider()
        st.markdown(
            f"""
            <div class="prediction-box {sentiment_class}">
                <h3><span style="font-weight: bold;">Sentiment</span>: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Please enter a review to analyze.")
