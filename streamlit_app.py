import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the JSON database
@st.cache_resource
def load_data():
    with open('agriculture_data.json') as f:
        return json.load(f)

# Initialize the summarization pipeline
@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Initialize the sentence transformer model for encoding
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode text using the sentence transformer model
def encode_texts(sentence_model, texts):
    return sentence_model.encode(texts)

# Search relevant data from the database using cosine similarity
def search_database(sentence_model, data, question):
    question_embedding = encode_texts(sentence_model, [question])[0]
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
    context_embeddings = encode_texts(sentence_model, context_texts)
    similarities = cosine_similarity([question_embedding], context_embeddings)[0]

    # Get the top 3 most similar entries
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_similarities = similarities[top_indices]
    relevant_context = "\n\n".join([context_entries[idx][0] for idx in top_indices])

    # Explainable AI Information
    explainable_info = (
        f"### Explainable AI Information:\n\n"
        f"Your question: **{question}**\n\n"
        f"The top 3 most relevant entries from the database were selected based on cosine similarity scores:\n\n"
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

# Format context for summarization
def format_context(context):
    return f"Here is the information about growing tomatoes:\n{context.replace('\n', '. ').replace('  ', ' ')}"

# Summarize context using the summarization model
def summarize_context(summarization_pipeline, context):
    formatted_context = format_context(context)
    summarized = summarization_pipeline(
        formatted_context,
        max_length=150,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return summarized[0]['summary_text']

# Main application
def main():
    st.title("Agriculture Information Database")
    data = load_data()
    summarization_pipeline = load_summarization_pipeline()
    sentence_model = load_sentence_model()

    st.header("Ask a Question")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if sentence_model and summarization_pipeline:
            context = search_database(sentence_model, data, user_question)
            if context.strip():
                st.write("**Context Provided to Model:**", context)
                summary = summarize_context(summarization_pipeline, context)
                st.write("**Summary:**", summary)
            else:
                st.write("No relevant information found in the database.")
        else:
            st.error("Model initialization failed.")

if __name__ == '__main__':
    main()
