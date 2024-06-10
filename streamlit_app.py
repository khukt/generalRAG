import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import psutil
import os
import json
import torch

# Load JSON database
@st.cache_resource
def load_json_database(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load crop data from JSON file
crop_data = load_json_database('crop_data.json')

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    return model, tokenizer

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
model, tokenizer = load_model()

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

embeddings = generate_embeddings(crop_data)

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

# Improved function to automatically determine question type
def determine_question_type(question):
    question = question.lower()
    if any(keyword in question for keyword in ["how", "grow", "plant", "cultivate"]):
        return "Planting Guide"
    elif any(keyword in question for keyword in ["issues", "problems", "diseases", "pests"]):
        return "Common Issues"
    elif any(keyword in question for keyword in ["best practices", "tips", "guidelines", "recommendations"]):
        return "Best Practices"
    elif any(keyword in question for keyword in ["watering", "irrigation", "water schedule"]):
        return "Watering Schedule"
    elif any(keyword in question for keyword in ["fertilization", "fertilizer", "feeding", "nutrition"]):
        return "Fertilization Tips"
    elif any(keyword in question for keyword in ["harvest", "harvesting", "pick", "picking"]):
        return "Harvest Timing"
    else:
        return "Planting Guide"  # Default to planting guide if no keywords match

# Function to load templates
def load_templates(file_path='templates.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {
            "Planting Guide": (
                "Please provide a detailed guide on how to plant and grow the specified crop based on the following question and context.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Guide:"
            ),
            "Common Issues": (
                "Please provide a detailed explanation of common issues and their solutions for growing the specified crop based on the following question and context.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Issues and Solutions:"
            ),
            "Best Practices": (
                "Please provide a detailed list of best practices for growing the specified crop based on the following question and context.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Best Practices:"
            ),
            "Watering Schedule": (
                "Please provide a detailed watering schedule for the specified crop based on the following question and context.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Watering Schedule:"
            ),
            "Fertilization Tips": (
                "Please provide detailed fertilization tips for the specified crop based on the following question and context.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Fertilization Tips:"
            ),
            "Harvest Timing": (
                "Please provide detailed harvest timing information for the specified crop based on the following question and context.\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Harvest Timing:"
            )
        }

# Function to save templates
def save_templates(templates, file_path='templates.json'):
    with open(file_path, 'w') as file:
        json.dump(templates, file, indent=4)

# Load existing templates or default ones
templates = load_templates()

# Function to generate text based on input question and context
def generate_paragraph(question_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    input_text = templates.get(question_type, templates["Planting Guide"]).format(question=question, context=context)
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

question = st.text_input("Question", value="How to grow tomatoes?", key="question")

if question:
    relevant_context = find_relevant_context(question, embeddings)
    context = generate_context("Crop", relevant_context)
    question_type = determine_question_type(question)
else:
    context = ""
    question_type = "Planting Guide"

st.subheader("Detected Question Type")
st.write(f"**{question_type}**")

st.subheader("Context")
st.markdown(f"```{context}```")

# Additional controls for model.generate parameters in the sidebar
st.sidebar.title("Model Parameters")
max_length = st.sidebar.slider("Max Length", 50, 500, 300)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5)
no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size", 1, 10, 2)
early_stopping = st.sidebar.checkbox("Early Stopping", value=True)

# Template configuration
st.sidebar.title("Template Configuration")
selected_question_type = st.sidebar.selectbox("Select Question Type", list(templates.keys()))

template_input = st.sidebar.text_area("Template", value=templates[selected_question_type])
if st.sidebar.button("Save Template"):
    templates[selected_question_type] = template_input
    save_templates(templates)
    st.sidebar.success("Template saved successfully!")

if question:
    with st.spinner("Generating..."):
        guide, memory_footprint = generate_paragraph(question_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping)
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
