import os
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv

# Store Hugging Face models in D drive
os.environ["HF_HOME"] = "D:/summarize_proj/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/summarize_proj/model_cache"
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
MAX_LENGTH = int(os.getenv("MAX_LENGTH"))
MIN_LENGTH = int(os.getenv("MIN_LENGTH"))

@st.cache_resource
def load_model():
    return pipeline("summarization", model=MODEL_NAME)

summarizer = load_model()
st.title("Document Summarization App 📄")
st.write("Paste your document below or upload a file.")

# MAIN INPUT (like ChatGPT)
text_input = st.text_area(
    "✍️ Enter your document",
    height=250,
    placeholder="Paste or type your document here..."
)

# Secondary option
uploaded_file = st.file_uploader("📂 Or upload a .txt file", type=["txt"])
text = ""

# Priority to manual text
if text_input.strip() != "":
    text = text_input
elif uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

if text:
    st.subheader("Document Preview 📜")
    st.write(text)
    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = summarizer(
                text[:3000],
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH,
                do_sample=False
            )
        st.subheader("Summary")
        st.write(summary[0]["summary_text"])