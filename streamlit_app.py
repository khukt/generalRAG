import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate text based on input question and context
def generate_paragraph(crop_name, question, context):
    input_text = (
        f"Please provide a detailed, step-by-step guide on how to grow {crop_name.lower()} based on the following question and context.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Steps:"
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_output(answer)

# Function to format the output into a well-written paragraph
def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences if sentence)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    return formatted_output

# Function to load context from a JSON file
@st.cache_resource
def load_context_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the JSON data (assuming the file is named 'crop_details.json')
context_data = load_context_from_json('crop_details.json')

# Function to get context text from the JSON data
def get_context_text(crop_name, context_data):
    crop = context_data.get(crop_name, {})
    if not crop:
        return "Crop details not found."
    context = (
        f"Crop Name: {crop.get('name', 'N/A')}\n"
        f"Planting Season: {crop.get('planting_season', 'N/A')}\n"
        f"Harvest Time: {crop.get('harvest_time', 'N/A')}\n"
        f"Soil Type: {crop.get('soil_type', 'N/A')}\n"
        f"Watering Needs: {crop.get('watering_needs', 'N/A')}\n"
        f"Pests and Diseases: {', '.join(crop.get('pests_diseases', []))}\n"
    )
    return context

# Streamlit UI
st.title("Crop Growing Guide Generator")
st.write("Enter your question and select a crop to generate a detailed guide.")

question = st.text_input("Question", value="How to grow this crop?")
crop_name = st.selectbox("Select Crop", options=list(context_data.keys()), index=0)
context = get_context_text(crop_name, context_data)

if st.button("Generate Guide"):
    with st.spinner("Generating..."):
        guide = generate_paragraph(crop_name, question, context)
    st.subheader("Generated Guide")
    st.write(guide)

# Display the selected crop context
st.subheader("Crop Context")
st.write(context)
