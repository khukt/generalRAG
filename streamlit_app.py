import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import psutil
import os
import json
import torch
import gc

# Function to clear previous model from memory
def clear_model_from_memory():
    if "model" in st.session_state:
        del st.session_state.model
    if "tokenizer" in st.session_state:
        del st.session_state.tokenizer
    torch.cuda.empty_cache()
    gc.collect()

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_model(model_name):
    clear_model_from_memory()
    if "t5" in model_name or "flan" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    elif "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    elif "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name)
        tokenizer = None  # No tokenizer needed for sentence-transformers
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    return model, tokenizer

# Load JSON database
@st.cache_resource
def load_json_database(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load crop data from JSON file
@st.cache_resource
def get_crop_data():
    return load_json_database('crop_data.json')

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# General function to generate context from details
def generate_context(key, details):
    context_lines = [f"{key.capitalize()}:"]
    for k, v in details.items():
        if isinstance(v, list):
            v = ', '.join(map(str, v))
        elif isinstance(v, dict):
            v = generate_context(k, v)  # Recursively handle nested dictionaries
        context_lines.append(f"{k.replace('_', ' ').title()}: {v}")
    return '\n'.join(context_lines)

# Generate embeddings for contexts in batches
@st.cache_resource
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
@st.cache_data
def find_relevant_context(question, _embeddings):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(list(_embeddings.values())))
    best_match_index = torch.argmax(cosine_scores).item()
    best_match_key = list(_embeddings.keys())[best_match_index]
    return crop_data[best_match_key], best_match_key

# Function to load templates
@st.cache_resource
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

# Function to perform paraphrasing
def paraphrase(model, tokenizer, sentence, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    if isinstance(model, SentenceTransformer):
        # Use the util.paraphrase_miniLM function for SentenceTransformer model
        paraphrases = util.paraphrase_miniLM([sentence])
        return paraphrases[0], 0
    else:
        input_text = f"paraphrase: {sentence}"
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
st.title("Paraphrasing Task")
st.write("Enter a sentence to generate its paraphrase.")

# Add a selectbox for model selection
model_name = st.selectbox(
    "Select Model",
    [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "gpt2"
    ],
    index=0
)

# Clear previous model cache if a new model is selected
if "previous_model_name" in st.session_state and st.session_state.previous_model_name != model_name:
    load_model.clear()
    clear_model_from_memory()

st.session_state.previous_model_name = model_name

crop_data = get_crop_data()
embedding_model = load_embedding_model()
model, tokenizer = load_model(model_name)
embeddings = generate_embeddings(crop_data)

sentence = st.text_input("Sentence", value="How to grow tomatoes?", key="sentence")

# Additional controls for model.generate parameters in the sidebar
st.sidebar.title("Model Parameters")
max_length = st.sidebar.slider("Max Length", 50, 500, 300)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5)
no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size", 1, 10, 2)
early_stopping = st.sidebar.checkbox("Early Stopping", value=True)

# Buttons to clear cache and reload models
st.sidebar.title("Cache Management")
if st.sidebar.button("Clear Cache and Reload Models"):
    load_model.clear()
    st.experimental_rerun()

if sentence:
    with st.spinner("Generating paraphrase..."):
        paraphrased_sentence, memory_footprint = paraphrase(model, tokenizer, sentence, max_length, num_beams, no_repeat_ngram_size, early_stopping)
    st.subheader("Generated Paraphrase")
    st.write(paraphrased_sentence)
    
    # Calculate total memory usage and other memory usage
    total_memory_usage = memory_usage()
    other_memory_usage = total_memory_usage - model_memory_usage - memory_footprint
    
    st.subheader("Memory Usage Details")
    st.write(f"Model memory usage: {model_memory_usage:.2f} MB")
    st.write(f"Memory used during generation: {memory_footprint:.2f} MB")
    st.write(f"Other memory usage: {other_memory_usage:.2f} MB")
    st.write(f"Total memory usage: {total_memory_usage:.2f} MB")

# Function to find the most relevant context based on the question
if sentence:
    relevant_context, best_match_key = find_relevant_context(sentence, embeddings)
    context = generate_context(best_match_key, relevant_context)
    st.subheader("Relevant Context")
    st.markdown(f"```{context}```")
