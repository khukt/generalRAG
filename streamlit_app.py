import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import json
import torch
import gc
import psutil
import os
import time

# Function to clear previous model from memory
def clear_model_from_memory():
    if "model" in st.session_state:
        del st.session_state.model
    if "tokenizer" in st.session_state:
        del st.session_state.tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Load the model and tokenizer with error handling
@st.cache_resource
def load_model(model_name):
    clear_model_from_memory()
    try:
        if model_name in ["google/flan-t5-small", "google/flan-t5-base"]:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        elif model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model {model_name}: {e}")
        return None, None

# Function to load JSON database
@st.cache_resource
def load_json_database(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to measure memory usage
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

# Function to measure computation time
def measure_time(start_time):
    return time.time() - start_time

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
model_options = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "facebook/bart-large",
    "gpt2"
]
model_name = st.selectbox("Select Model", model_options, index=1)
model, tokenizer = load_model(model_name)

# Add a button to clear model from memory
if st.button('Clear Model from Memory'):
    clear_model_from_memory()

if model and tokenizer:
    # Step 1: User question input
    st.subheader("Step 1: User Question Input")
    question = st.text_input("Question", value="How to grow tomatoes?", key="question")
    step1_memory = memory_usage()
    step1_time = measure_time(time.time())
    st.write(f"Memory Usage: {step1_memory:.2f} MB")
    st.write(f"Computation Time: {step1_time:.2f} seconds")

    if question:
        # Step 2: Load crop data from JSON file
        st.subheader("Step 2: Load Crop Data from JSON File")
        crop_data = load_json_database('crop_data.json')
        step2_memory = memory_usage()
        step2_time = measure_time(time.time())
        st.write(f"Memory Usage: {step2_memory:.2f} MB")
        st.write(f"Computation Time: {step2_time:.2f} seconds")

        # Step 3: Generate embeddings for contexts
        st.subheader("Step 3: Generate Embeddings for Contexts")
        embeddings = generate_embeddings(crop_data)
        step3_memory = memory_usage()
        step3_time = measure_time(time.time())
        st.write(f"Memory Usage: {step3_memory:.2f} MB")
        st.write(f"Computation Time: {step3_time:.2f} seconds")

        # Step 4: Compute cosine similarity between question and contexts
        st.subheader("Step 4: Compute Cosine Similarity")
        question_embedding = load_embedding_model().encode(question, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(list(embeddings.values())))
        best_match_index = torch.argmax(cosine_scores).item()
        relevant_crop = list(crop_data.keys())[best_match_index]
        relevant_crop_details = crop_data[relevant_crop]
        step4_memory = memory_usage()
        step4_time = measure_time(time.time())
        st.write(f"Memory Usage: {step4_memory:.2f} MB")
        st.write(f"Computation Time: {step4_time:.2f} seconds")

        if relevant_crop_details:
            # Step 5: Generate context from relevant crop details
            st.subheader("Step 5: Generate Context from Relevant Crop Details")
            context = generate_context(relevant_crop, relevant_crop_details)
            st.write(f"```{context}```")
            step5_memory = memory_usage()
            step5_time = measure_time(time.time())
            st.write(f"Memory Usage: {step5_memory:.2f} MB")
            st.write(f"Computation Time: {step5_time:.2f} seconds")

            # Step 6: Generate paragraph using the model
            st.subheader("Step 6: Generate Guide")
            with st.spinner("Generating guide..."):
                guide = generate_paragraph(model, tokenizer, question, context, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
            st.write(guide)
            step6_memory = memory_usage()
            step6_time = measure_time(time.time())
            st.write(f"Memory Usage: {step6_memory:.2f} MB")
            st.write(f"Computation Time: {step6_time:.2f} seconds")

            # Step 7: Display total memory usage and computation time
            st.subheader("Step 7: Memory Usage and Computation Time Details")
            total_memory_usage = memory_usage()
            total_computation_time = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time
            st.write(f"Total Memory Usage: {total_memory_usage:.2f} MB")
            st.write(f"Total Computation Time: {total_computation_time:.2f} seconds")
        else:
            st.write("No relevant crop found in the database for the given question.")
else:
    st.write("Model could not be loaded. Please try selecting a different model or check for errors.")
