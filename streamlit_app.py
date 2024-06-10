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
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
model, tokenizer = load_model()

# Function to generate embeddings for crop contexts
@st.cache_resource
def generate_crop_embeddings(crop_data):
    embeddings = {}
    for crop, details in crop_data.items():
        context = generate_context(details)
        embeddings[crop] = embedding_model.encode(context, convert_to_tensor=True)
    return embeddings

def generate_context(details):
    context = []
    for key, value in details.items():
        if isinstance(value, list):
            value = ', '.join(value)
        context.append(f"{key.replace('_', ' ').title()}: {value}")
    return '\n'.join(context)

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Measure memory usage after loading the model
model_memory_usage = memory_usage()

# Function to find the most relevant crop context based on the question
@st.cache_data
def find_relevant_crop_context(question, _crop_embeddings):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(list(_crop_embeddings.values())))
    best_match_index = torch.argmax(cosine_scores).item()
    best_match_crop = list(_crop_embeddings.keys())[best_match_index]
    return crop_data[best_match_crop]

# Function to determine question type
def determine_question_type(question):
    keyword_mapping = {
        "Step-by-Step Guide": ["how", "grow", "plant", "steps", "step-by-step"],
        "Common Issues": ["issues", "problems", "diseases", "pests"],
        "Best Practices": ["best practices", "tips", "guidelines", "recommendations"],
        "Watering Schedule": ["watering", "irrigation", "water schedule"],
        "Fertilization Tips": ["fertilization", "fertilizer", "feeding", "nutrition"],
        "Harvest Timing": ["harvest", "harvesting", "pick", "picking"]
    }
    
    question = question.lower()
    for question_type, keywords in keyword_mapping.items():
        if any(keyword in question for keyword in keywords):
            return question_type
    return "Step-by-Step Guide"  # Default to step-by-step if no keywords match

# Function to get the template for the given question type
def get_template(question_type):
    templates = {
        "Step-by-Step Guide": "Please provide a detailed, step-by-step guide on how to grow the specified crop based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nSteps:",
        "Common Issues": "Please provide a detailed explanation of common issues and their solutions for growing the specified crop based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nIssues and Solutions:",
        "Best Practices": "Please provide a detailed list of best practices for growing the specified crop based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nBest Practices:",
        "Watering Schedule": "Please provide a detailed watering schedule for the specified crop based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nWatering Schedule:",
        "Fertilization Tips": "Please provide detailed fertilization tips for the specified crop based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nFertilization Tips:",
        "Harvest Timing": "Please provide detailed harvest timing information for the specified crop based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nHarvest Timing:",
    }
    return templates.get(question_type, templates["Step-by-Step Guide"])

# Function to generate text based on input question and context
def generate_paragraph(question_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    template = get_template(question_type)
    input_text = template.format(question=question, context=context)
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
    crop_embeddings = generate_crop_embeddings(crop_data)
    relevant_context = find_relevant_crop_context(question, crop_embeddings)
    context = generate_context(relevant_context)
    question_type = determine_question_type(question)
else:
    context = ""
    question_type = "Step-by-Step Guide"

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
