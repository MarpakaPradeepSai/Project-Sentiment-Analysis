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

# --- Function to get background color based on sentiment (slightly adjusted colors) ---
def get_background_color(label):
    if "Positive" in label:
        return "#E0F7FA"  # Softer, lighter cyan for positive
    elif "Neutral" in label:
        return "#FFFDE7"  # Softer, lighter yellow for neutral
    else:
        return "#FFEBEE"  # Softer, lighter red for negative

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
    /* Import Google Fonts - Using Poppins and Roboto for a modern feel */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700&family=Roboto:wght@400;500&display=swap');

    .main {
        background-color: #FAFAFA; /* Lighter background */
        font-family: 'Roboto', sans-serif; /* Modern body font */
        color: #333;
    }
    h1 {
        font-family: 'Poppins', sans-serif; /* Modern title font */
        color: #4A148C; /* Deeper purple for title */
        text-align: center;
        font-size: 2.8em; /* Slightly reduced title size */
        margin-bottom: 15px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2); /* Softer text shadow */
        font-weight: 700; /* Bolder title */
    }
    .stButton>button {
        background: linear-gradient(90deg, #00BCD4, #673AB7); /* Modernized button gradient */
        color: white !important;
        border: none;
        border-radius: 20px; /* Slightly less rounded buttons */
        padding: 12px 24px; /* Adjusted button padding */
        font-size: 1.1em;
        font-weight: 500; /* Slightly lighter button font weight */
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03); /* Slightly less scale on hover */
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2); /* Softer box shadow on hover */
        color: white !important;
    }
    .prediction-box {
        border-radius: 20px; /* Adjusted prediction box border-radius */
        padding: 15px; /* Adjusted prediction box padding */
        text-align: center;
        font-size: 1.1em; /* Slightly larger prediction box font */
        border: 1px solid #E0E0E0; /* Light border for prediction box */
        margin-top: 15px; /* Add margin above prediction box */
    }
    .stTextArea textarea {
        border-radius: 12px; /* Adjusted text area border-radius */
        border: 1px solid #CED4DA;
        padding: 12px; /* Adjusted text area padding */
        background-color: #FFFFFF;
        box-shadow: inset 2px 2px 5px #F0F0F0; /* Softer inset shadow for text area */
    }
    .stTextArea label { /* Style for the text area label */
        font-weight: 500;
        margin-bottom: 5px;
        display: block;
        color: #555; /* Darker label text */
    }
    .stTextArea textarea::placeholder {
        color: #AAA; /* Even lighter placeholder text */
        font-style: italic;
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

cols = st.columns(5)
for i, url in enumerate(image_urls):
    with cols[i]:
        st.image(url, width=100)

# --- Main Content Area ---
with st.container(): # Using container to group input and output
    col1, col2 = st.columns([3, 1]) # Adjust column widths as needed

    with col1:
        user_input = st.text_area("Enter your AirPods review here") # Label is now placeholder

    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🔍 Analyze Sentiment"): # Button in the second column
            if user_input:
                with st.spinner('Analyzing sentiment...'): # Keep spinner
                    time.sleep(0.5) # Simulate processing time, remove in real use if fast enough
                    sentiment_probs = predict_sentiment(user_input)
                    sentiment_label = get_sentiment_label(sentiment_probs[0])
                    background_color = get_background_color(sentiment_label)

                st.divider() # Keep divider
                st.markdown(
                    f"""
                    <div style="background-color:{background_color};" class="prediction-box">
                        <h3><span style="font-weight: 500;">Sentiment</span>: {sentiment_label}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("⚠️ Please enter a review to analyze.") # Keep warning message with emoji
