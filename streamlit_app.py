import streamlit as st
import json
from transformers import pipeline

# Load the JSON database
with open('agriculture_data.json') as f:
    data = json.load(f)

# Initialize the transformer pipeline for question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Title of the app
st.title("Agriculture Information Database")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a query type", ["Crop Information", "Pest and Disease Management", "Ask a Question"])

# Function to get crop information
def get_crop_info(crop_name):
    for crop in data['crops']:
        if crop['name'].lower() == crop_name.lower():
            return crop
    return None

# Function to get pest and disease information
def get_pest_disease_info(name):
    for pest_disease in data['pests_diseases']:
        if pest_disease['name'].lower() == name.lower():
            return pest_disease
    return None

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
        # Determine relevant context
        relevant_context = ""

        # Check if the question is about a specific crop
        if "grow" in user_question.lower() or "plant" in user_question.lower() or "harvest" in user_question.lower():
            for crop in data['crops']:
                if crop['name'].lower() in user_question.lower():
                    relevant_context += f"Crop Name: {crop['name']}\n"
                    relevant_context += f"Planting Season: {crop['planting_season']}\n"
                    relevant_context += f"Harvest Time: {crop['harvest_time']}\n"
                    relevant_context += f"Soil Type: {crop['soil_type']}\n"
                    relevant_context += f"Watering Needs: {crop['watering_needs']}\n"
                    relevant_context += f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n"

        # Check if the question is about pests or diseases
        if "pest" in user_question.lower() or "disease" in user_question.lower() or "treat" in user_question.lower() or "symptom" in user_question.lower():
            for pest in data['pests_diseases']:
                if pest['name'].lower() in user_question.lower():
                    relevant_context += f"Pest/Disease Name: {pest['name']}\n"
                    relevant_context += f"Affected Crops: {', '.join(pest['affected_crops'])}\n"
                    relevant_context += f"Symptoms: {pest['symptoms']}\n"
                    relevant_context += f"Treatment: {pest['treatment']}\n"

        if relevant_context:
            qa_result = qa_pipeline(question=user_question, context=relevant_context)
            st.write(f"**Answer:** {qa_result['answer']}")
        else:
            st.write("No relevant information found in the database.")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
