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
    sentiment_mapping = ["Negative 😞", "Neutral 😐", "Positive 😊"]
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# --- Function to get background color based on sentiment (softer colors) ---
def get_background_color(label):
    if "Positive" in label:
        return "#E0F7FA"  # Very Light Blue
    elif "Neutral" in label:
        return "#FFFDE7"  # Very Light Yellow
    else:
        return "#FFEBEE"  # Very Light Red

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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700&family=Open+Sans:wght@400;600&display=swap');

    .main {
        background-color: #f8f9fa; /* Very light gray background */
        font-family: 'Open Sans', sans-serif;
        color: #333; /* Dark gray text */
    }
    h1 {
        font-family: 'Nunito', sans-serif;
        color: #2c3e50; /* Darker blue-gray for title */
        text-align: center;
        font-size: 2.8em;
        margin-bottom: 15px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(to right, #4CAF50, #8BC34A); /* Green gradient button */
        color: white !important;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.15);
        transition: transform 0.1s ease-in-out, box-shadow 0.1s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(to right, #66BB6A, #9CCC65); /* Lighter green on hover */
    }
    .prediction-box {
        border-radius: 20px;
        padding: 15px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        margin-top: 15px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stTextArea textarea {
        border-radius: 15px;
        border: 1px solid #ced4da;
        padding: 12px;
        background-color: #fff;
        box-shadow: inset 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .stTextArea textarea::placeholder {
        color: #999; /* Light gray placeholder text */
        font-style: italic;
    }
    """,
    unsafe_allow_html=True
)

# --- App Title and Subheader ---
st.markdown("<h1 style='margin-top: 0;'>AirPods Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("A simple tool to analyze the sentiment of your Apple AirPods reviews.")

# --- AirPods Image Row ---
image_urls = [
    "https://i5.walmartimages.com/seo/Apple-AirPods-with-Charging-Case-2nd-Generation_8540ab4f-8062-48d0-9133-323a99ed921d.fb43fa09a0faef3f9495feece1397f8d.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/b6247579-386a-4bda-99aa-01e44801bc33.49db04f5e5b8d7f329c6580455e2e010.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/0f803868-d25f-4891-b0c8-e27a514ede02.f22c42c1ea17cd4d2b30fdfc89a8797c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/df1b081f-4fa9-4ea5-87f8-413b9cad7a6e.f580d742da0a58bc25dadd30512adf72.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF",
    "https://i5.walmartimages.com/asr/2830c8d7-292d-4b99-b92f-239b15ff1062.ce77d20b2f20a569bfd656d05ca89f7c.jpeg?odnHeight=117&odnWidth=117&odnBg=FFFFFF"
]

cols = st.columns(5)
for i, url in enumerate(image_urls):
    with cols[i]:
        st.image(url, width=100)

# --- User Input Text Area ---
user_input = st.text_area("**Write your review of Apple AirPods below:**", placeholder="Enter your AirPods review here")

# --- Analyze Sentiment Button ---
if st.button("✨ Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing sentiment...'): # Show spinner while processing
            time.sleep(0.5) # Simulate processing time, remove in real use if fast enough
            sentiment_probs = predict_sentiment(user_input)
            sentiment_label = get_sentiment_label(sentiment_probs[0])
            background_color = get_background_color(sentiment_label)

        st.divider() # Visual separator before result
        st.markdown(
            f"""
            <div style="background-color:{background_color};" class="prediction-box">
                <h3>Sentiment: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a review to analyze.")
