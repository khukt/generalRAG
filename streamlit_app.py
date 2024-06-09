import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the JSON database
try:
    with open('agriculture_data.json') as f:
        data = json.load(f)
    st.write("Data loaded successfully.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = {}

# Initialize the summarization pipeline (using t5-small)
try:
    summarization_pipeline = pipeline("summarization", model="t5-small")
    st.write("Summarization model loaded successfully.")
except Exception as e:
    st.error(f"Error loading summarization model: {e}")
    summarization_pipeline = None

# Initialize the sentence transformer model for encoding
try:
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    st.write("Sentence transformer model loaded successfully.")
except Exception as e:
    st.error(f"Error loading sentence transformer model: {e}")
    sentence_model = None

# Title of the app
st.title("Agriculture Information Database")

# Function to encode text using the sentence transformer model
def encode_texts(texts):
    try:
        return sentence_model.encode(texts)
    except Exception as e:
        st.error(f"Error encoding texts: {e}")
        return []

# Function to search relevant data from the database using cosine similarity
def search_database(question):
    try:
        question_embedding = encode_texts([question])[0]
        context_entries = []

        for crop in data.get('crops', []):
            crop_text = f"Crop Name: {crop['name']}\n"
            crop_text += f"Planting Season: {crop['planting_season']}\n"
            crop_text += f"Harvest Time: {crop['harvest_time']}\n"
            crop_text += f"Soil Type: {crop['soil_type']}\n"
            crop_text += f"Watering Needs: {crop['watering_needs']}\n"
            crop_text += f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n"
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
        explainable_info = "### Explainable AI Information:\n\n"
        explainable_info += f"Your question: **{question}**\n\n"
        explainable_info += "The top 3 most relevant entries from the database were selected based on cosine similarity scores:\n\n"

        for idx, sim in zip(top_indices, top_similarities):
            explainable_info += f"- **Entry Index**: {idx}\n"
            explainable_info += f"  - **Crop Name**: {context_entries[idx][1]}\n"
            explainable_info += f"  - **Cosine Similarity Score**: {sim:.4f}\n"
            explainable_info += f"  - **Details**: {context_entries[idx][0]}\n\n"

        st.markdown(explainable_info)
        st.write("Relevant Context Generated:", relevant_context)

        return relevant_context
    except Exception as e:
        st.error(f"Error searching database: {e}")
        return ""

# Function to summarize context using the summarization model
def summarize_context(context):
    try:
        input_text = f"Summarize the following information about growing tomatoes:\n{context}\nSummary:"
        summarized = summarization_pipeline(
            input_text, 
            max_length=150, 
            min_length=50, 
            num_beams=4, 
            length_penalty=2.0, 
            early_stopping=True
        )
        return summarized[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing context: {e}")
        return ""

# Ask a Question Page
st.header("Ask a Question")
user_question = st.text_input("Enter your question:")
if st.button("Ask"):
    try:
        if sentence_model:
            # Search the database for relevant context
            context = search_database(user_question)
            if context.strip():
                st.write("**Context Provided to Model:**", context)
                if summarization_pipeline:
                    # Summarize the context
                    summary = summarize_context(context)
                    st.write("**Summary:**", summary)
                else:
                    st.error("Summarization pipeline is not initialized.")
            else:
                st.write("No relevant information found in the database.")
        else:
            st.error("Sentence transformer model is not initialized.")
    except Exception as e:
        st.error(f"Error during processing: {e}")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
