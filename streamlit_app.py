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

# Function to search relevant data from the database
def search_database(question):
    relevant_context = ""

    question_lower = question.lower()

    # Search for relevant crop information
    for crop in data['crops']:
        if crop['name'].lower() in question_lower:
            relevant_context += f"How to grow {crop['name']}:\n"
            relevant_context += f"Planting Season: {crop['planting_season']}\n"
            relevant_context += f"Harvest Time: {crop['harvest_time']}\n"
            relevant_context += f"Soil Type: {crop['soil_type']}\n"
            relevant_context += f"Watering Needs: {crop['watering_needs']}\n"
            relevant_context += f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n\n"

    return relevant_context

# Function to post-process the model's answer
def post_process_answer(answer, question):
    if not answer.strip() or answer.strip().lower() == "soil":
        return "I couldn't find the specific information you were looking for. Please try rephrasing your question or provide more details."
    return f"Based on your question about '{question}', here is the information:\n\n{answer.strip()}"

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
        context = search_database(user_question)
        if context:
            qa_result = qa_pipeline(question=user_question, context=context)
            answer = post_process_answer(qa_result['answer'], user_question)
            st.write(f"**Answer:** {answer}")
        else:
            st.write("No relevant information found in the database.")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
