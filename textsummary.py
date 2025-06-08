import streamlit as st
from transformers import pipeline
from rouge_score import rouge_scorer

# Load summarization model
summarizer = pipeline("summarization")

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("📝 Text Summarizer Web")
st.write("Paste your text below and get a summary instantly.")


# Text input
text_input = st.text_area("Enter your text here:", height=300)

# Reference summary input (optional)
reference = st.text_area("Enter reference (gold) summary for comparison (optional):", height=150)

# Summarize button
if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        max_input_length = 1024
        if len(text_input.split()) > max_input_length:
            st.info(f"Trimming input to first {max_input_length} tokens...")
            text_input = " ".join(text_input.split()[:max_input_length])

        with st.spinner("Summarizing..."):
            summary = summarizer(text_input, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
            st.success("Summary:")
            st.write(summary)

            if reference.strip():
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(reference, summary)
                st.subheader("📊 ROUGE Scores (vs reference):")
                for key in scores:
                    st.write(f"**{key.upper()}**: Precision = {scores[key].precision:.2f}, Recall = {scores[key].recall:.2f}, F1 = {scores[key].fmeasure:.2f}")
