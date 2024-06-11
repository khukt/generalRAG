import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import psutil
import os
import json
import torch
import gc
import time

# Helper function to measure and log execution time of a function
def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        st.write(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @log_time
    def load_model(self, model_name):
        self.clear_model_from_memory()
        if "t5" in model_name or "flan" in model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        elif "gpt2" in model_name:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        st.session_state.model = self.model
        st.session_state.tokenizer = self.tokenizer

    @log_time
    def clear_model_from_memory(self):
        if "model" in st.session_state:
            del st.session_state.model
        if "tokenizer" in st.session_state:
            del st.session_state.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

class TemplateManager:
    def __init__(self, template_file='templates.json'):
        self.template_file = template_file
        self.templates = self.load_templates()

    @log_time
    def load_templates(self):
        if os.path.exists(self.template_file):
            with open(self.template_file, 'r') as file:
                return json.load(file)
        else:
            return self.default_templates()

    def save_templates(self):
        with open(self.template_file, 'w') as file:
            json.dump(self.templates, file, indent=4)

    def default_templates(self):
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

    def get_templates(self):
        return self.templates

    def update_template(self, question_type, template, keywords):
        self.templates[question_type]["template"] = template
        self.templates[question_type]["keywords"] = [keyword.strip() for keyword in keywords.split(',')]
        self.save_templates()

class CropDataManager:
    def __init__(self):
        self.data = load_crop_data()

    def get_crop_data(self):
        return self.data

@st.cache_resource
def load_crop_data():
    return load_json_database('crop_data.json')

@st.cache_resource
def load_json_database(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class EmbeddingManager:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.embeddings = None

    def get_embeddings(self, data):
        if self.embeddings is None:
            self.embeddings = generate_embeddings(self.embedding_model, data)
        return self.embeddings

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def generate_embeddings(embedding_model, data):
    keys = list(data.keys())
    contexts = [generate_context(key, data[key]) for key in keys]
    context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)
    return dict(zip(keys, context_embeddings))

def generate_context(key, details):
    context_lines = []
    for k, v in details.items():
        if isinstance(v, list):
            v = ', '.join(map(str, v))
        elif isinstance(v, dict):
            v = generate_context(k, v)  # Recursively handle nested dictionaries
        context_lines.append(f"{k.replace('_', ' ').title()}: {v}")
    return '\n'.join(context_lines)

@log_time
def find_relevant_context(question, embeddings, data):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(list(embeddings.values())))
    best_match_index = torch.argmax(cosine_scores).item()
    best_match_key = list(embeddings.keys())[best_match_index]
    return data[best_match_key]

def determine_question_type(question, templates):
    question = question.lower()
    for question_type, details in templates.items():
        if any(keyword in question for keyword in details.get("keywords", [])):
            return question_type
    return "Planting Guide"  # Default to planting guide if no keywords match

@log_time
def generate_text(model, tokenizer, task_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping, use_template, templates, question_type):
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

def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences if sentence)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    return formatted_output

# Streamlit UI
st.title("Crop Growing Guide Generator")
st.write("Enter your question to generate a detailed guide.")

# Initialize managers
model_manager = ModelManager()
template_manager = TemplateManager()
crop_data_manager = CropDataManager()
embedding_manager = EmbeddingManager()

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
selected_question_type = st.sidebar.selectbox("Select Question Type", list(template_manager.get_templates().keys()))

template_input = st.sidebar.text_area("Template", value=template_manager.get_templates()[selected_question_type]["template"])
keywords_input = st.sidebar.text_area("Keywords (comma separated)", value=", ".join(template_manager.get_templates()[selected_question_type]["keywords"]))
if st.sidebar.button("Save Template"):
    template_manager.update_template(selected_question_type, template_input, keywords_input)
    st.sidebar.success("Template saved successfully!")

# Buttons to clear cache and reload models, embeddings, and templates
st.sidebar.title("Cache Management")
if st.sidebar.button("Clear Cache and Reload Models"):
    model_manager.clear_model_from_memory()
    st.experimental_rerun()

if st.sidebar.button("Clear Cache and Reload Data"):
    load_crop_data.clear()
    generate_embeddings.clear()
    st.experimental_rerun()

if st.sidebar.button("Clear Cache and Reload Templates"):
    template_manager.load_templates()
    st.experimental_rerun()

# Main input and processing section
crop_data = crop_data_manager.get_crop_data()
embedding_model = embedding_manager.embedding_model
model_manager.load_model(model_name)
model, tokenizer = model_manager.get_model_and_tokenizer()
embeddings = embedding_manager.get_embeddings(crop_data)

question = st.text_input("Question", value="How to grow tomatoes?", key="question")

if question:
    relevant_context = find_relevant_context(question, embeddings, crop_data)
    context = generate_context("Crop", relevant_context)
    question_type = determine_question_type(question, template_manager.get_templates())
else:
    context = ""
    question_type = "Planting Guide"

st.subheader("Detected Question Type")
st.write(f"**{question_type}**")

st.subheader("Context")
st.markdown(f"```{context}```")

if question:
    with st.spinner("Generating..."):
        guide, memory_footprint = generate_text(model, tokenizer, task_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping, use_template, template_manager.get_templates(), question_type)
    st.subheader("Generated Guide")
    st.write(guide)
    
    # Calculate total memory usage and other memory usage
    total_memory_usage = memory_usage()
    other_memory_usage = total_memory_usage - memory_footprint
    
    st.subheader("Memory Usage Details")
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
