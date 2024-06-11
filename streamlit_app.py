import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import psutil
import os
import json
import torch
import gc

# Function to clear the model and tokenizer from memory
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
    context_lines = []
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
    return crop_data[best_match_key]

# Function to automatically determine the question type
def determine_question_type(question, templates):
    question = question.lower()
    for question_type, details in templates.items():
        if any(keyword in question for keyword in details.get("keywords", [])):
            return question_type
    return "Planting Guide"  # Default to planting guide if no keywords match

# Function to load templates from a JSON file
@st.cache_resource
def load_templates(file_path='templates.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {
            "Planting Guide": {
                "template": (
                    "Provide a guide on planting and growing the specified crop.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Guide:"
                ),
                "keywords": ["how", "grow", "plant", "cultivate"]
            },
            "Common Issues": {
                "template": (
                    "Explain common issues and their solutions for growing the specified crop.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Issues and Solutions:"
                ),
                "keywords": ["issues", "problems", "diseases", "pests"]
            },
            "Best Practices": {
                "template": (
                    "List the best practices for growing the specified crop.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Best Practices:"
                ),
                "keywords": ["best practices", "tips", "guidelines", "recommendations"]
            },
            "Watering Schedule": {
                "template": (
                    "Provide a watering schedule for the specified crop.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Watering Schedule:"
                ),
                "keywords": ["watering", "irrigation", "water schedule"]
            },
            "Fertilization Tips": {
                "template": (
                    "Provide fertilization tips for the specified crop.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Fertilization Tips:"
                ),
                "keywords": ["fertilization", "fertilizer", "feeding", "nutrition"]
            },
            "Harvest Timing": {
                "template": (
                    "Provide harvest timing information for the specified crop.\n\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Harvest Timing:"
                ),
                "keywords": ["harvest", "harvesting", "pick", "picking"]
            }
        }

# Function to save templates to a JSON file
def save_templates(templates, file_path='templates.json'):
    with open(file_path, 'w') as file:
        json.dump(templates, file, indent=4)

# Load existing templates or default ones
templates = load_templates()

# Function to generate text based on input question and context
def generate_text(model, tokenizer, task_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping, use_template):
    # Determine input text based on task type and template usage
    input_text = ""
    if use_template:
        input_text = templates[question_type]["template"].format(question=question, context=context)
    else:
        input_text = f"{context} {question}"

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
st.write("Enter your question to generate a detailed guide.")

# Sidebar for model selection and parameters
st.sidebar.title("Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "distilgpt2",
        "gpt2"
    ],
    index=1
)

if "t5" in model_name or "flan" in model_name:
    task_type = st.sidebar.selectbox(
        "Select Task",
        [
            "Generation",
            "Paraphrasing",
            "Summarization"
        ],
        index=0
    )
else:
    task_type = st.sidebar.selectbox(
        "Select Task",
        [
            "Generation",
            "Paraphrasing",
            "Continuation"
        ],
        index=0
    )

use_template = st.sidebar.checkbox("Use Template", value=True)

# Additional controls for model.generate parameters in the sidebar
st.sidebar.title("Model Parameters")
max_length = st.sidebar.slider("Max Length", 50, 500, 300)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5)
no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size", 1, 10, 2)
early_stopping = st.sidebar.checkbox("Early Stopping", value=True)

# Template configuration
st.sidebar.title("Template Configuration")
selected_question_type = st.sidebar.selectbox("Select Question Type", list(templates.keys()))

template_input = st.sidebar.text_area("Template", value=templates[selected_question_type]["template"])
keywords_input = st.sidebar.text_area("Keywords (comma separated)", value=", ".join(templates[selected_question_type]["keywords"]))
if st.sidebar.button("Save Template"):
    templates[selected_question_type]["template"] = template_input
    templates[selected_question_type]["keywords"] = [keyword.strip() for keyword in keywords_input.split(',')]
    save_templates(templates)
    st.sidebar.success("Template saved successfully!")

# Buttons to clear cache and reload models, embeddings, and templates
st.sidebar.title("Cache Management")
if st.sidebar.button("Clear Cache and Reload Models"):
    load_model.clear()
    load_embedding_model.clear()
    generate_embeddings.clear()
    st.experimental_rerun()

if st.sidebar.button("Clear Cache and Reload Data"):
    get_crop_data.clear()
    generate_embeddings.clear()
    st.experimental_rerun()

if st.sidebar.button("Clear Cache and Reload Templates"):
    load_templates.clear()
    st.experimental_rerun()

# Main input and processing section
crop_data = get_crop_data()
embedding_model = load_embedding_model()
model, tokenizer = load_model(model_name)
embeddings = generate_embeddings(crop_data)

question = st.text_input("Question", value="How to grow tomatoes?", key="question")

if question:
    relevant_context = find_relevant_context(question, embeddings)
    context = generate_context("Crop", relevant_context)
    question_type = determine_question_type(question, templates)
else:
    context = ""
    question_type = "Planting Guide"

st.subheader("Detected Question Type")
st.write(f"**{question_type}**")

st.subheader("Context")
st.markdown(f"```{context}```")

if question:
    with st.spinner("Generating..."):
        guide, memory_footprint = generate_text(model, tokenizer, task_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping, use_template)
    st.subheader("Generated Guide")
    st.write(guide)
    
    # Calculate total memory usage and other memory usage
    total_memory_usage = memory_usage()
    other_memory_usage = total_memory_usage - model_memory_usage - memory_footprint
    
    st.subheader("Memory Usage Details")
    st.write(f"Model memory usage: {model_memory_usage:.2f} MB")
    st.write(f"Memory used during generation: {memory_footprint:.2f} MB")
    st.write(f"Other memory usage: {other_memory_usage:.2f} MB")
    st.write(f"Total memory usage: {total_memory_usage:.2f} MB")

    # Detailed memory breakdown
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    st.write(f"Resident Set Size (RSS): {mem_info.rss / (1024 ** 2):.2f} MB")
    st.write(f"Virtual Memory Size (VMS): {mem_info.vms / (1024 ** 2):.2f} MB")
    st.write(f"Shared Memory: {mem_info.shared / (1024 ** 2):.2f} MB")
    st.write(f"Text (Code): {mem_info.text / (1024 ** 2):.2f} MB")
    st.write(f"Data + Stack: {mem_info.data / (1024 ** 2):.2f} MB")
    st.write(f"Library (unused): {mem_info.lib / (1024 ** 2):.2f} MB")
