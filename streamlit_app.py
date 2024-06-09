import streamlit as st
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the JSON database
with open('agriculture_data.json') as f:
    data = json.load(f)

# Initialize the text generation model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

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

# Function to preprocess the question and generate context
def generate_context(question):
    context = ""
    question_lower = question.lower()
    
    # Determine relevant crop information
    if "grow" in question_lower or "plant" in question_lower or "harvest" in question_lower:
        for crop in data['crops']:
            if crop['name'].lower() in question_lower:
                context += f"How to grow {crop['name']}:\n"
                context += f"Planting Season: {crop['planting_season']}\n"
                context += f"Harvest Time: {crop['harvest_time']}\n"
                context += f"Soil Type: {crop['soil_type']}\n"
                context += f"Watering Needs: {crop['watering_needs']}\n"
                context += f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n\n"
    
    # Determine relevant pest/disease information
    if "pest" in question_lower or "disease" in question_lower or "treat" in question_lower or "symptom" in question_lower:
        for pest in data['pests_diseases']:
            if pest['name'].lower() in question_lower:
                context += f"Information about {pest['name']}:\n"
                context += f"Affected Crops: {', '.join(pest['affected_crops'])}\n"
                context += f"Symptoms: {pest['symptoms']}\n"
                context += f"Treatment: {pest['treatment']}\n\n"
    
    # Include more comprehensive information
    for crop in data['crops']:
        context += f"General information about growing {crop['name']}:\n"
        context += f"Planting Season: {crop['planting_season']}\n"
        context += f"Harvest Time: {crop['harvest_time']}\n"
        context += f"Soil Type: {crop['soil_type']}\n"
        context += f"Watering Needs: {crop['watering_needs']}\n"
        context += f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n\n"
    
    return context

# Function to post-process the model's answer
def post_process_answer(answer):
    return answer.strip()

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
        context = generate_context(user_question)
        if context:
            prompt = f"Question: {user_question}\nContext: {context}\nAnswer:"
            generated_answers = text_generation_pipeline(prompt, max_length=200, num_return_sequences=1)
            generated_answer = generated_answers[0]['generated_text']
            answer = post_process_answer(generated_answer.replace(prompt, "").strip())
            st.write(f"**Answer:** {answer}")
        else:
            st.write("No relevant information found in the database.")

if __name__ == '__main__':
    st.write("Welcome to the Agriculture Information Database!")
