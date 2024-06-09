import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the JSON database
def load_data():
    try:
        with open('agriculture_data.json') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}

data = load_data()

# Initialize the summarization pipeline (using DistilBART)
def init_summarization_pipeline():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None

summarization_pipeline = init_summarization_pipeline()

# Initialize the sentence transformer model for encoding
def init_sentence_transformer():
    try:
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading sentence transformer model: {e}")
        return None

sentence_model = init_sentence_transformer()

# Title of the app
st.title("Agriculture Information Database")

# Function to encode text using the sentence transformer model
def encode_texts(texts):
    return sentence_model.encode(texts)

# Function to search relevant data from the database using cosine similarity
def search_database(question):
    question_embedding = encode_texts([question])[0]
    context_entries = []

    for crop in data.get('crops', []):
        crop_text = (
            f"Crop Name: {crop['name']}\n"
            f"Planting Season: {crop['planting_season']}\n"
            f"Harvest Time: {crop['harvest_time']}\n"
            f"Soil Type: {crop['soil_type']}\n"
            f"Watering Needs: {crop['watering_needs']}\n"
            f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n"
        )
        context_entries.append((crop_text, crop['name']))

    # Calculate similarities
    context_texts = [entry[0] for entry in context_entries]
    context_embeddings = encode_texts(context_texts)
    similarities = cosine_similarity([question_embedding], context_embeddings)[0]

    # Get the top 3 most similar entries
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_similarities = similarities[top_indices]
    relevant_context = "\n\n".join([context_entries[idx][0] for idx in top_indices])

    # Explainable AI Information
    explainable_info = (
        "### Explainable AI Information:\n\n"
        f"Your question: **{question}**\n\n"
        "The top 3 most relevant entries from the database were selected based on cosine similarity scores:\n\n"
    )

    for idx, sim in zip(top_indices, top_similarities):
        explainable_info += (
            f"- **Entry Index**: {idx}\n"
            f"  - **Crop Name**: {context_entries[idx][1]}\n"
            f"  - **Cosine Similarity Score**: {sim:.4f}\n"
            f"  - **Details**: {context_entries[idx][0]}\n\n"
        )

    st.markdown(explainable_info)
    st.write("Relevant Context Generated:", relevant_context)

    return relevant_context

# Function to summarize context using the summarization model
def summarize_context(context):
    try:
        summarized = summarization_pipeline(context, max_length=100, min_length=30, do_sample=False)
        return summarized[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

# Ask a Question Page
st.header("Ask a Question")
user_question = st.text_input("Enter your question:")
if st.button("Ask"):
    if sentence_model and summarization_pipeline:
        # Search the database for relevant context
        context = search_database(user_question)
        if context.strip():
            st.write("**Context Provided to Model:**", context)
            # Summarize the context using the summarization model
            answer = summarize_context(context)
            st.write("**Answer:**", answer)
        else:
            st.write("No relevant information found in the database.")
    else:
        st.error("Sentence transformer model or summarization pipeline is not initialized.")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
