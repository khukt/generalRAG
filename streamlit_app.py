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

# Specify the path to the JSON data file
json_file_path = 'crop_data.json'  # Change this to the path of your JSON file

# Load crop data from JSON file
crop_data = load_json_database(json_file_path)

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

# Function to generate embeddings for data contexts
@st.cache_resource
def generate_embeddings(data):
    embeddings = {}
    for key, details in data.items():
        context = generate_context(details)
        embeddings[key] = embedding_model.encode(context, convert_to_tensor=True)
    return embeddings

# Function to generate context from details
def generate_context(details):
    if isinstance(details, dict):
        context = []
        for key, value in details.items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            elif isinstance(value, dict):
                value = generate_context(value)  # Recursively handle nested dictionaries
            context.append(f"{key.replace('_', ' ').title()}: {value}")
        return {key.replace('_', ' ').title(): value for key, value in details.items()}
    else:
        return {details: details}

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

# Function to determine question type
def determine_question_type(question, keyword_mapping):
    question = question.lower()
    for question_type, keywords in keyword_mapping.items():
        if any(keyword in question for keyword in keywords):
            return question_type
    return "General Information"  # Default if no keywords match

# Function to get the template for the given question type
def get_template(question_type, template_mapping):
    return template_mapping.get(question_type, template_mapping["General Information"])

# Function to generate text based on input question and context
def generate_paragraph(template, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    input_text = template.format(question=question, **context)
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
st.title("Generalized Data Guide Generator")
st.write("Enter your question to generate a detailed guide.")

# Load embeddings once after the app starts
embeddings = generate_embeddings(crop_data)

question = st.text_input("Question", value="How to grow tomatoes?", key="question")

st.sidebar.title("Keyword and Template Configuration")

# User-defined keyword and template mappings
keyword_mapping = {}
template_mapping = {}

st.sidebar.subheader("Define question types and keywords")
question_types = st.sidebar.text_area(
    "Format: Type: keyword1, keyword2, ...",
    value="Step-by-Step Guide: how, steps, process, guide\n"
          "Common Issues: issues, problems, errors, troubleshoot\n"
          "Best Practices: best practices, tips, recommendations, guidelines\n"
          "Watering Schedule: watering, irrigation, water schedule\n"
          "Fertilization Tips: fertilization, fertilizer, feeding, nutrition\n"
          "Harvest Timing: harvest, timing, pick, picking\n"
          "General Information: information, details, overview"
)

for line in question_types.split('\n'):
    if ':' in line:
        q_type, keywords = line.split(':', 1)
        keyword_mapping[q_type.strip()] = [k.strip() for k in keywords.split(',')]

st.sidebar.subheader("Define templates for question types")
templates = st.sidebar.text_area(
    "Format: Type: Template",
    value="Step-by-Step Guide: To grow tomatoes, follow these steps based on the provided context. Planting season: {Planting Season}. Prepare the soil by {Soil Preparation}. Ensure the soil type is {Soil Type}. Water regularly, keeping the soil moist but not waterlogged. Fertilize every {Fertilization Schedule}. Manage pests such as {Pests Diseases} by {Pest Management}. Harvest tomatoes when they are {Harvesting Techniques}.\n"
          "Common Issues: For growing tomatoes, common issues include {Pests Diseases}. To manage these issues, you can {Pest Management}.\n"
          "Best Practices: The best practices for growing tomatoes include {Fertilization Schedule}, {Soil Preparation}, and managing pests by {Pest Management}.\n"
          "Watering Schedule: Watering schedule for tomatoes is {Watering Frequency}.\n"
          "Fertilization Tips: Fertilize tomatoes every {Fertilization Schedule} with a balanced fertilizer.\n"
          "Harvest Timing: Harvest tomatoes {Harvesting Techniques}.\n"
          "General Information: Here is some general information about growing tomatoes: Planting season: {Planting Season}, Soil type: {Soil Type}, Harvest time: {Harvest Time}.\n"
)

for line in templates.split('\n'):
    if ':' in line:
        q_type, template = line.split(':', 1)
        template_mapping[q_type.strip()] = template.strip()

if question:
    relevant_context = find_relevant_context(question, embeddings)
    context = generate_context(relevant_context)
    question_type = determine_question_type(question, keyword_mapping)
else:
    context = ""
    question_type = "General Information"

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
        template = get_template(question_type, template_mapping)
        guide, memory_footprint = generate_paragraph(template, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping)
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
