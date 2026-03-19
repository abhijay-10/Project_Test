import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
MAX_LENGTH = int(os.getenv("MAX_LENGTH"))
MIN_LENGTH = int(os.getenv("MIN_LENGTH"))

# Load summarization model
summarizer = pipeline("summarization", model=MODEL_NAME)

st.title("Document Summarization App")

uploaded_file = st.file_uploader("Upload a text document", type=["txt"])

if uploaded_file is not None:

    text = uploaded_file.read().decode("utf-8")

    st.subheader("Document Content")
    st.write(text)

    if st.button("Summarize Document"):

        with st.spinner("Generating summary..."):

            summary = summarizer(
                text,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH,
                do_sample=False
            )

        st.subheader("Summary")
        st.write(summary[0]["summary_text"])