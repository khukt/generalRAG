import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import torch
import gc
import psutil
import os

# Sample JSON data
crop_data_json = '''
{
  "Tomato": {
    "name": "Tomato",
    "planting_season": "Spring",
    "harvest_time": "Summer",
    "soil_type": "Well-drained, fertile soil",
    "soil_preparation": "Till the soil and add compost before planting.",
    "watering_frequency": "Regular watering, keep soil moist but not waterlogged.",
    "fertilization_schedule": "Fertilize every 2 weeks with a balanced fertilizer.",
    "pests_diseases": ["Aphids", "Blight", "Tomato Hornworm"],
    "pest_management": "Use insecticidal soap for aphids and handpick tomato hornworms.",
    "harvesting_techniques": "Harvest when tomatoes are firm and fully colored."
  },
  "Corn": {
    "name": "Corn",
    "planting_season": "Late Spring",
    "harvest_time": "Late Summer to Early Fall",
    "soil_type": "Well-drained, loamy soil",
    "soil_preparation": "Mix in aged manure or compost before planting.",
    "watering_frequency": "Moderate watering, keep soil moist especially during tasseling and ear development.",
    "fertilization_schedule": "Side-dress with nitrogen fertilizer when plants are 8 inches tall and again when tassels appear.",
    "pests_diseases": ["Corn Earworm", "Rootworm", "Corn Smut"],
    "pest_management": "Use Bacillus thuringiensis (Bt) for earworms and rotate crops to manage rootworms.",
    "harvesting_techniques": "Harvest when ears are full and kernels are milky when punctured."
  }
}
'''

# Parse JSON data
crop_data = json.loads(crop_data_json)

# Function to clear previous model from memory
def clear_model_from_memory():
    if "model" in st.session_state:
        del st.session_state.model
    if "tokenizer" in st.session_state:
        del st.session_state.tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# Load the model and tokenizer
@st.cache_resource
def load_model(model_name):
    clear_model_from_memory()
    if "t5" in model_name or "flan" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    return model, tokenizer

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Function to generate context from JSON data
def generate_context(crop_name, crop_details):
    context_lines = [f"{crop_name.capitalize()}:"]
    for key, value in crop_details.items():
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        context_lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return '\n'.join(context_lines)

# Function to generate text based on input question and context
def generate_paragraph(model, tokenizer, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs, max_length=max_length, num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit UI
st.title("GraphRAG Enhanced Crop Growing Guide Generator")
st.write("Enter your question to generate a detailed guide using GraphRAG.")

# Model selection
model_name = st.selectbox("Select Model", ["google/flan-t5-small", "google/flan-t5-base"], index=1)
model, tokenizer = load_model(model_name)

# User question input
question = st.text_input("Question", value="How to grow tomatoes?", key="question")

if question:
    # Identify relevant crop based on question (simple heuristic)
    relevant_crop = None
    for crop in crop_data.keys():
        if crop.lower() in question.lower():
            relevant_crop = crop
            break

    if relevant_crop:
        context = generate_context(relevant_crop, crop_data[relevant_crop])
        st.subheader("Generated Context")
        st.write(f"```{context}```")

        with st.spinner("Generating guide..."):
            guide = generate_paragraph(model, tokenizer, question, context, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        st.subheader("Generated Guide")
        st.write(guide)

        # Memory usage details
        total_memory_usage = memory_usage()
        st.subheader("Memory Usage Details")
        st.write(f"Total memory usage: {total_memory_usage:.2f} MB")
    else:
        st.write("No relevant crop found in the database for the given question.")
