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
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = {}

# Initialize the transformer pipeline for question answering
try:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
except Exception as e:
    st.error(f"Error loading transformer model: {e}")
    qa_pipeline = None

# Initialize the sentence transformer model for encoding
try:
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error loading sentence transformer model: {e}")
    sentence_model = None

# Title of the app
st.title("Agriculture Information Database")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a query type", ["Crop Information", "Pest and Disease Management", "Ask a Question"])

# Function to get crop information
def get_crop_info(crop_name):
    for crop in data.get('crops', []):
        if crop['name'].lower() == crop_name.lower():
            return crop
    return None

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

    return relevant_context

# Function to post-process the model's answer
def post_process_answer(answer, question):
    if not answer.strip() or answer.strip().lower() == "soil":
        return "I couldn't find the specific information you were looking for. Please try rephrasing your question or provide more details."
    return f"Based on your question about '{question}', here is the information:\n\n{answer.strip()}"

# Function to format context for better readability
def format_context(context):
    formatted_context = ""
    lines = context.split("\n")
    for line in lines:
        if "Crop Name" in line:
            formatted_context += f"\n**{line}**\n"
        elif "Planting Season" in line or "Harvest Time" in line or "Soil Type" in line or "Watering Needs" in line or "Pests and Diseases" in line:
            formatted_context += f"- {line}\n"
    return formatted_context

# Crop Information Page
if option == "Crop Information":
    st.header("Crop Information")
    crop_name = st.text_input("Enter the crop name:")
    if st.button("Search"):
        crop_info = get_crop_info(crop_name)
        if crop_info:
            st.write(f"**Name:** {crop_info['name']}")
            st.write(f"**Planting Season:** {crop_info['planting_season']}")
            st.write(f"**Harvest Time:** {crop_info['harvest_time']}")
            st.write(f"**Soil Type:** {crop_info['soil_type']}")
            st.write(f"**Watering Needs:** {crop_info['watering_needs']}")
            st.write(f"**Pests and Diseases:** {', '.join(crop_info['pests_diseases'])}")
        else:
            st.error("Crop not found!")

# Pest and Disease Management Page
if option == "Pest and Disease Management":
    st.header("Pest and Disease Management")
    pest_disease_name = st.text_input("Enter the pest or disease name:")
    if st.button("Search"):
        pest_disease_info = get_pest_disease_info(pest_disease_name)
        if pest_disease_info:
            st.write(f"**Name:** {pest_disease_info['name']}")
            st.write(f"**Affected Crops:** {', '.join(pest_disease_info['affected_crops'])}")
            st.write(f"**Symptoms:** {pest_disease_info['symptoms']}")
            st.write(f"**Treatment:** {pest_disease_info['treatment']}")
        else:
            st.error("Pest or disease not found!")

# Ask a Question Page
if option == "Ask a Question":
    st.header("Ask a Question")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if sentence_model:
            context = search_database(user_question)
            if context.strip():
                formatted_context = format_context(context)
                st.write("**Context Provided to Model:**", formatted_context)  # Debugging line
                if qa_pipeline:
                    qa_result = qa_pipeline(question=user_question, context=formatted_context)
                    st.write("**QA Pipeline result:**", qa_result)  # Debugging line
                    answer = post_process_answer(qa_result['answer'], user_question)
                    st.write(f"**Answer:** {answer}")
                else:
                    st.error("QA pipeline is not initialized.")
            else:
                st.write("No relevant information found in the database.")
        else:
            st.error("Sentence transformer model is not initialized.")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
