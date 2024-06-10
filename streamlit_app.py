import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import psutil
import os

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Function to generate text based on input question and context
def generate_paragraph(question_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    templates = {
        "step-by-step": (
            f"Please provide a detailed, step-by-step guide on how to grow the specified crop based on the following question and context.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Steps:"
        ),
        "common issues": (
            f"Please provide a detailed explanation of common issues and their solutions for growing the specified crop based on the following question and context.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Issues and Solutions:"
        ),
        "best practices": (
            f"Please provide a detailed list of best practices for growing the specified crop based on the following question and context.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Best Practices:"
        )
    }
    
    input_text = templates.get(question_type, templates["step-by-step"])
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Measure memory before generation
    memory_before = memory_usage()
    
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_beams=num_beams, 
        no_repeat_ngram_size=no_repeat_ngram_size, 
        early_stopping=early_stopping
    )
    
    # Measure memory after generation
    memory_after = memory_usage()
    
    memory_footprint = memory_after - memory_before
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_output(answer), memory_footprint

# Function to format the output into a well-written paragraph
def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences if sentence)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    return formatted_output

# Streamlit UI
st.title("Crop Growing Guide Generator")
st.write("Select a crop, question type, and enter your question and context to generate a detailed guide.")

crop_choice = st.selectbox("Select Crop", ["Tomato", "Corn"])

crop_contexts = {
    "Tomato": """
        Crop Name: Tomato
        Planting Season: Spring
        Harvest Time: Summer
        Soil Type: Well-drained, fertile soil
        Watering Needs: Regular watering, keep soil moist but not waterlogged
        Pests and Diseases: Aphids, Blight, Tomato Hornworm
    """,
    "Corn": """
        Crop Name: Corn
        Planting Season: Late Spring
        Harvest Time: Late Summer to Early Fall
        Soil Type: Well-drained, loamy soil
        Watering Needs: Moderate watering, keep soil moist especially during tasseling and ear development
        Pests and Diseases: Corn Earworm, Rootworm, Corn Smut
    """
}

question_type = st.selectbox("Select Question Type", ["step-by-step", "common issues", "best practices"])
question = st.text_input("Question", value=f"How to grow {crop_choice.lower()}?")
context = st.text_area("Context", value=crop_contexts[crop_choice])

# Additional controls for model.generate parameters
max_length = st.slider("Max Length", 50, 500, 300)
num_beams = st.slider("Number of Beams", 1, 10, 5)
no_repeat_ngram_size = st.slider("No Repeat N-Gram Size", 1, 10, 2)
early_stopping = st.checkbox("Early Stopping", value=True)

if st.button("Generate Guide"):
    with st.spinner("Generating..."):
        guide, memory_footprint = generate_paragraph(question_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping)
    st.subheader("Generated Guide")
    st.write(guide)
    st.subheader("Memory Footprint")
    st.write(f"Memory used during generation: {memory_footprint:.2f} MB")

# Cache resource decorator for efficient reloading
@st.cache_resource
def get_crop_details(crop_name):
    if crop_name == "Tomato":
        return {
            'name': 'Tomato',
            'planting_season': 'Spring',
            'harvest_time': 'Summer',
            'soil_type': 'Well-drained, fertile soil',
            'watering_needs': 'Regular watering, keep soil moist but not waterlogged',
            'pests_diseases': ['Aphids', 'Blight', 'Tomato Hornworm']
        }
    else:
        return {
            'name': 'Corn',
            'planting_season': 'Late Spring',
            'harvest_time': 'Late Summer to Early Fall',
            'soil_type': 'Well-drained, loamy soil',
            'watering_needs': 'Moderate watering, keep soil moist especially during tasseling and ear development',
            'pests_diseases': ['Corn Earworm', 'Rootworm', 'Corn Smut']
        }

crop_details = get_crop_details(crop_choice)
crop_text = (
    f"Crop Name: {crop_details['name']}\n"
    f"Planting Season: {crop_details['planting_season']}\n"
    f"Harvest Time: {crop_details['harvest_time']}\n"
    f"Soil Type: {crop_details['soil_type']}\n"
    f"Watering Needs: {crop_details['watering_needs']}\n"
    f"Pests and Diseases: {', '.join(crop_details['pests_diseases'])}\n"
)
st.write(crop_text)
