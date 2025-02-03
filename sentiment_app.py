import streamlit as st
from transformers import AlbertTokenizer, AutoModelForSequenceClassification
import torch
import requests
import os
import time  # For adding a loading spinner
import random # For random emoji placement

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

# --- Function to get background color based on sentiment (original colors) ---
def get_background_color(label):
    if "Positive" in label:
        return "#C3E6CB"  # Original softer green
    elif "Neutral" in label:
        return "#FFE8A1"  # Original softer yellow
    else:
        return "#F5C6CB"  # Original softer red

# --- Function to get floating emojis based on sentiment ---
def get_floating_emojis(label):
    if "Positive" in label:
        emojis = ["🎉", "🌟", "🎈", "👍"] # More positive emojis for floating effect
    elif "Neutral" in label:
        emojis = ["🤔", "💭", "🧘"] # Neutral/thinking emojis
    else:
        emojis = ["💔", "😔", "😞", "👎", "😠"] # More negative emojis
    return emojis

# --- Function to generate HTML for floating emojis ---
def generate_floating_emoji_html(emojis, sentiment_label):
    emoji_spans = ""
    for emoji in emojis:
        # Randomize initial position and animation delay for each emoji
        start_x = random.uniform(10, 90) # Random horizontal start position within the box
        start_y = random.uniform(10, 90) # Random vertical start position within the box
        animation_delay = random.uniform(0, 2) # Random delay to stagger animation

        emoji_spans += f"""
            <span class="floating-emoji" style="
                left: {start_x}%;
                top: {start_y}%;
                animation-delay: {animation_delay}s;
            ">{emoji}</span>
        """

    background_color = get_background_color(sentiment_label)
    return f"""
        <div style="position: relative; background-color:{background_color}; padding: 20px; border-radius: 25px; text-align: center; overflow: hidden;" class="prediction-box">
            <h3><span style="font-weight: bold;">Sentiment</span>: {sentiment_label}</h3>
            <div class="emoji-container">
                {emoji_spans}
            </div>
        </div>
    """


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
        background-color: #F0F2F6;
        font-family: 'Open Sans', sans-serif;
        color: #333;
    }
    h1 {
        font-family: 'Nunito', sans-serif;
        color: #6a0572;
        text-align: center;
        font-size: 3em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white !important;
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
        color: white !important;
    }
    .prediction-box {
        border-radius: 25px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
        position: relative; /* Needed for absolute positioning of emojis */
        overflow: hidden; /* Clip emojis if they go outside the box */
    }
    .stTextArea textarea {
        border-radius: 15px;
        border: 1px solid #ced4da;
        padding: 10px;
        background-color: #FFFFFF;
        box-shadow: 3px 3px 5px #9E9E9E;
    }
    .stTextArea textarea::placeholder {
        color: #999;
        font-style: italic;
    }

    .emoji-container {
        position: absolute; /* Container to hold emojis relative to prediction-box */
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none; /* Make sure emojis don't interfere with clicks */
    }

    .floating-emoji {
        position: absolute; /* Float emojis within emoji-container */
        font-size: 20px; /* Adjust emoji size */
        animation: floatEmoji 3s linear infinite; /* Apply floating animation */
        opacity: 0.8; /* Make emojis slightly transparent */
        pointer-events: none; /* Prevent emoji from being interactive */
    }

    @keyframes floatEmoji {
        0% {
            transform: translateY(0) rotate(0deg);
            opacity: 0;
        }
        10%, 90% {
            opacity: 0.8;
        }
        50% {
            transform: translateY(-10px) rotate(20deg); /* Adjust float height and rotation */
        }
        100% {
            transform: translateY(0) rotate(0deg);
            opacity: 0;
        }
    }


    """,
    unsafe_allow_html=True
)

# --- App Title ---
st.markdown(
    """
    <h1 style="font-size: 45px; text-align: center;">Apple AirPods Sentiment Analysis</h1>
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

# --- User Input Text Area ---
user_input = st.text_area("Enter your AirPods review here") # Original placeholder, removed bold label

# --- Analyze Sentiment Button ---
if st.button("🔍 Analyze Sentiment"): # Original button text and icon
    if user_input:
        with st.spinner('Analyzing sentiment...'): # Keep spinner
            time.sleep(0.5) # Simulate processing time, remove in real use if fast enough
            sentiment_probs = predict_sentiment(user_input)
            sentiment_label = get_sentiment_label(sentiment_probs[0])
            floating_emojis = get_floating_emojis(sentiment_label)
            floating_emoji_html = generate_floating_emoji_html(floating_emojis, sentiment_label)


        st.divider() # Keep divider
        st.markdown(floating_emoji_html, unsafe_allow_html=True)

    else:
        st.error("⚠️ Please enter a review to analyze.") # Keep warning message with emoji
