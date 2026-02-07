import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time  # For adding a loading spinner

# --- Set page config MUST be the first Streamlit command ---
st.set_page_config(
    page_title="AirPods Review Sentiment Analyzer",
    page_icon=":headphones:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Hugging Face Model Repository ---
model_name = "IamPradeep/Apple-Airpods-Sentiment-Analysis-ALBERT-base-v2"

# --- Load tokenizer and model from Hugging Face Hub ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

try:
    with st.spinner('Loading model from Hugging Face Hub...'):
        tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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

# --- Streamlit app ---


# --- Custom CSS for a more attractive look ---
st.markdown(
    """
    <style>
    /* Import Google Fonts - Keeping Nunito and Open Sans for general text */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700&family=Open+Sans:wght@400;600&display=swap');

    .main {
        background-color: #F0F2F6; /* Original main background color */
        font-family: 'Open Sans', sans-serif; /* Keep Open Sans for body */
        color: #333;
    }
    h1 {
        font-family: 'Nunito', sans-serif; /* Keep Nunito for title */
        color: #6a0572; /* Original title color */
        text-align: center;
        font-size: 3em; /* Original title size */
        margin-bottom: 15px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); /* Original text shadow */
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71); /* Original button gradient */
        color: white !important;
        border: none;
        border-radius: 25px; /* Original button border-radius */
        padding: 10px 20px;
        font-size: 1.2em; /* Original button font-size */
        font-weight: bold; /* Original button font-weight */
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease; /* Original button transition */
    }
    .stButton>button:hover {
        transform: scale(1.05); /* Original button hover transform */
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); /* Original button hover box-shadow */
        color: white !important;
    }
    .prediction-box {
        border-radius: 25px; /* Original prediction box border-radius */
        padding: 10px; /* Original prediction box padding */
        text-align: center; /* Original prediction box text-align */
        font-size: 18px; /* Original prediction box font-size */
    }
    .stTextArea textarea {
        border-radius: 15px; /* Keep text area border-radius */
        border: 1px solid #ced4da; /* Keep text area border */
        padding: 10px; /* Keep text area padding */
        background-color: #FFFFFF; /* Keep text area background */
        box-shadow: 3px 3px 5px #9E9E9E; /* Keep text area shadow */
    }
    .stTextArea textarea::placeholder {
        color: #999; /* Light gray placeholder text - keep if desired */
        font-style: italic; /* Italic placeholder text - keep if desired */
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
            background_color = get_background_color(sentiment_label)

        st.divider() # Keep divider
        st.markdown(
            f"""
            <div style="background-color:{background_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3><span style="font-weight: bold;">Sentiment</span>: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Please enter a review to analyze.") # Keep warning message with emoji
