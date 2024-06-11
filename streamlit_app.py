import streamlit as st
from sentence_transformers import SentenceTransformer, util
import psutil
import os
import json
import torch
import gc

# Function to clear memory and cache
def clear_memory_cache():
    if "model" in st.session_state:
        del st.session_state.model
    torch.cuda.empty_cache()
    gc.collect()

# Function to load the model
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    clear_memory_cache()
    model = SentenceTransformer(model_name)
    return model

# Function to load JSON database
@st.cache_resource(show_spinner=False)
def load_json_database(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to load crop data
@st.cache_resource(show_spinner=False)
def get_crop_data():
    return load_json_database('crop_data.json')

# Function to load embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate context from details
def generate_context(key, details):
    context_lines = [f"{key.capitalize()}:"]
    for k, v in details.items():
        if isinstance(v, list):
            v = ', '.join(map(str, v))
        elif isinstance(v, dict):
            v = generate_context(k, v)  # Recursively handle nested dictionaries
        context_lines.append(f"{k.replace('_', ' ').title()}: {v}")
    return '\n'.join(context_lines)

# Function to generate embeddings for contexts
@st.cache_resource(show_spinner=False)
def generate_embeddings(data):
    keys = list(data.keys())
    contexts = [generate_context(key, data[key]) for key in keys]
    context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)
    return dict(zip(keys, context_embeddings))

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Measure memory usage after loading the model
model_memory_usage = memory_usage()

# Function to find the most relevant context based on the question
@st.cache_data(show_spinner=False)
def find_relevant_context(question, _embeddings):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(list(_embeddings.values())))
    best_match_index = torch.argmax(cosine_scores).item()
    best_match_key = list(_embeddings.keys())[best_match_index]
    return crop_data[best_match_key], best_match_key

# Function to load templates
@st.cache_resource(show_spinner=False)
def load_templates(file_path='templates.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {
            "Planting Guide": {
                "template": (
                    "Please provide a detailed guide on how to plant and grow the specified crop based on the following question and context.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Guide:"
                ),
                "keywords": ["how", "grow", "plant", "cultivate"]
            },
            "Common Issues": {
                "template": (
                    "Please provide a detailed explanation of common issues and their solutions for growing the specified crop based on the following question and context.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Issues and Solutions:"
                ),
                "keywords": ["issues", "problems", "diseases", "pests"]
            },
            "Best Practices": {
                "template": (
                    "Please provide a detailed list of best practices for growing the specified crop based on the following question and context.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Best Practices:"
                ),
                "keywords": ["best practices", "tips", "guidelines", "recommendations"]
            },
            "Watering Schedule": {
                "template": (
                    "Please provide a detailed watering schedule for the specified crop based on the following question and context.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Watering Schedule:"
                ),
                "keywords": ["watering", "irrigation", "water schedule"]
            },
            "Fertilization Tips": {
                "template": (
                    "Please provide detailed fertilization tips for the specified crop based on the following question and context.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Fertilization Tips:"
                ),
                "keywords": ["fertilization", "fertilizer", "feeding", "nutrition"]
            },
            "Harvest Timing": {
                "template": (
                    "Please provide detailed harvest timing information for the specified crop based on the following question and context.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Harvest Timing:"
                ),
                "keywords": ["harvest", "harvesting", "pick", "picking"]
            }
        }

# Function to save templates
def save_templates(templates, file_path='templates.json'):
    with open(file_path, 'w') as file:
        json.dump(templates, file, indent=4)

# Load existing templates or default ones
templates = load_templates()

# Function to perform paraphrasing using SentenceTransformer
def paraphrase(model, sentence):
    # Generate paraphrases using sentence embeddings and cosine similarity
    paraphrases = util.paraphrase_mining(model, [sentence], top_k=5)
    paraphrased_sentence = paraphrases[0][2]
    return paraphrased_sentence

# Streamlit UI
st.title("Paraphrasing Task")
st.write("Enter a sentence to generate its paraphrase.")

# Set model name for SentenceTransformer
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Clear previous model cache if a new model is selected
if "previous_model_name" in st.session_state and st.session_state.previous_model_name != model_name:
    st.cache_data.clear()
    st.cache_resource.clear()
    clear_memory_cache()

st.session_state.previous_model_name = model_name

crop_data = get_crop_data()
embedding_model = load_embedding_model()
model = load_model(model_name)
embeddings = generate_embeddings(crop_data)

sentence = st.text_input("Sentence", value="How to grow tomatoes?", key="sentence")

# Buttons to clear cache and reload models
st.sidebar.title("Cache Management")
if st.sidebar.button("Clear Cache and Reload Models"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

if sentence:
    with st.spinner("Generating paraphrase..."):
        paraphrased_sentence = paraphrase(model, sentence)
    st.subheader("Generated Paraphrase")
    st.write(paraphrased_sentence)
    
    # Calculate total memory usage and other memory usage
    total_memory_usage = memory_usage()
    st.subheader("Memory Usage Details")
    st.write(f"Total memory usage: {total_memory_usage:.2f} MB")

# Function to find the most relevant context based on the question
if sentence:
    relevant_context, best_match_key = find_relevant_context(sentence, embeddings)
    context = generate_context(best_match_key, relevant_context)
    st.subheader("Relevant Context")
    st.markdown(f"```{context}```")
    
    # Paraphrase the relevant context
    with st.spinner("Paraphrasing context..."):
        paraphrased_context = paraphrase(model, context)
    st.subheader("Paraphrased Context")
    st.write(paraphrased_context)
