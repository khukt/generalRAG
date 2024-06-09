import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the JSON database
try:
    with open('agriculture_data.json') as f:
        data = json.load(f)
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = {}

# Initialize the sentence transformer model for encoding
try:
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading sentence transformer model: {e}")
    sentence_model = None

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
        crop_text = f"Crop Name: {crop['name']}\nPlanting Season: {crop['planting_season']}\nHarvest Time: {crop['harvest_time']}\nSoil Type: {crop['soil_type']}\nWatering Needs: {crop['watering_needs']}\nPests and Diseases: {', '.join(crop['pests_diseases'])}\n"
        context_entries.append((crop_text, crop_text))

    # Calculate similarities
    context_texts = [entry[0] for entry in context_entries]
    context_embeddings = encode_texts(context_texts)
    similarities = cosine_similarity([question_embedding], context_embeddings)[0]

    # Get the top 3 most similar entries
    top_indices = np.argsort(similarities)[-3:][::-1]
    relevant_context = "\n\n".join([context_entries[idx][1] for idx in top_indices])

    # Debugging information
    st.write("Cosine Similarity Scores:", similarities[0])
    st.write("Top 3 Similar Entries' Indices:", top_indices[0])
    st.write("Relevant Context Generated:", relevant_context)

    return relevant_context

# Ask a Question Page
st.header("Ask a Question")
user_question = st.text_input("Enter your question:")
if st.button("Ask"):
    if sentence_model:
        context = search_database(user_question)
        if context.strip():
            st.write("**Relevant Information:**", context)
        else:
            st.write("No relevant information found in the database.")
    else:
        st.error("Sentence transformer model is not initialized.")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
