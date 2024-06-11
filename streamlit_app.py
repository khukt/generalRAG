import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import psutil
import os
import json
import torch
import gc
from networkx import Graph
import networkx as nx
from functools import wraps

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Decorator to track memory usage of cached functions
def track_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mem_before = memory_usage()
        result = func(*args, **kwargs)
        mem_after = memory_usage()
        st.write(f"Memory usage for {func.__name__}: {mem_after - mem_before:.2f} MB")
        return result
    return wrapper

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
@track_memory_usage
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
@track_memory_usage
def load_json_database(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load crop data from JSON file
@st.cache_resource
@track_memory_usage
def get_crop_data():
    return load_json_database('crop_data.json')

# Load embedding model
@st.cache_resource
@track_memory_usage
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
@track_memory_usage
def generate_embeddings(data):
    keys = list(data.keys())
    contexts = [generate_context(key, data[key]) for key in keys]
    context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)
    return dict(zip(keys, context_embeddings))

# Function to create a Knowledge Graph from the crop data
def create_knowledge_graph(data):
    G = Graph()
    for key, details in data.items():
        G.add_node(key, type='crop')
        for k, v in details.items():
            if isinstance(v, list):
                for item in v:
                    G.add_edge(key, item, type=k)
            elif isinstance(v, dict):
                for sub_key, sub_val in v.items():
                    G.add_edge(key, sub_key, type=k)
                    G.add_node(sub_key, type='sub-entity')
                    if isinstance(sub_val, list):
                        for item in sub_val:
                            G.add_edge(sub_key, item, type=sub_key)
                    else:
                        G.add_edge(sub_key, sub_val, type=sub_key)
            else:
                G.add_edge(key, v, type=k)
    return G

# Function to find relevant context using the Knowledge Graph
def find_relevant_context_graph(question, graph):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    nodes = list(graph.nodes(data=True))
    node_embeddings = embedding_model.encode([n[0] for n in nodes], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, node_embeddings)
    best_match_index = torch.argmax(cosine_scores).item()
    best_match_key = nodes[best_match_index][0]
    return crop_data[best_match_key]

# Improved function to automatically determine question type
def determine_question_type(question, templates):
    question = question.lower()
    for question_type, details in templates.items():
        if any(keyword in question for keyword in details.get("keywords", [])):
            return question_type
    return "Planting Guide"  # Default to planting guide if no keywords match

# Function to load templates
@st.cache_resource
@track_memory_usage
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

# Function to generate text based on input question and context
def generate_paragraph(model, tokenizer, question_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    input_text = templates[question_type]["template"].format(question=question, context=context)
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
