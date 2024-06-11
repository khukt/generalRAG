import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    st.rerun()  # Rerun the Streamlit app to ensure the model is fully cleared

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

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for contexts in batches
@st.cache_resource
def generate_embeddings(data):
    keys = list(data.keys())
    contexts = [generate_context(key, data[key]) for key in keys]
    embedding_model = load_embedding_model()
    context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)
    return dict(zip(keys, context_embeddings))

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
st.title("Optimized Crop Growing Guide Generator")
st.write("Enter your question to generate a detailed guide.")

# Model selection
model_name = st.selectbox("Select Model", ["google/flan-t5-small", "google/flan-t5-base"], index=1)
model, tokenizer = load_model(model_name)

# Add a button to clear model from memory
if st.button('Clear Model from Memory'):
    clear_model_from_memory()

# User question input
question = st.text_input("Question", value="How to grow tomatoes?", key="question")

if question:
    # Retrieve relevant context using embeddings
    embedding_model = load_embedding_model()
    embeddings = generate_embeddings(crop_data)
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(list(embeddings.values())))
    best_match_index = torch.argmax(cosine_scores).item()
    relevant_crop = list(crop_data.keys())[best_match_index]
    relevant_crop_details = crop_data[relevant_crop]

    if relevant_crop_details:
        context = generate_context(relevant_crop, relevant_crop_details)
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
