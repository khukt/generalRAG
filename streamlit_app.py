import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
import json
import psutil
import logging
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import signal
import plotly.graph_objs as go

MAX_MEMORY_MB = 2600  # Maximum memory usage allowed in MB

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

def check_memory_usage():
    memory_used = get_memory_usage()
    if memory_used > MAX_MEMORY_MB:
        st.warning("Memory usage exceeded the maximum limit. Restarting the application to free up memory.")
        os.kill(os.getpid(), signal.SIGUSR1)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility functions
def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

def log_decision(message):
    logger.info(message)

def log_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        initial_memory = memory_usage()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        final_memory = memory_usage()
        memory_used = final_memory - initial_memory
        logger.info(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds, Memory used: {memory_used:.2f} MB")
        return result
    return wrapper

def log_model_usage(model_name):
    logger.info(f"Model used: {model_name}")

def log_question(question):
    logger.info(f"Question received: {question}")

def log_generation_details(details):
    logger.info(f"Generation details: {details}")

def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences if sentence)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    return formatted_output

def normalize_attention_weights(attentions):
    last_layer_attentions = attentions[-1]  # Get the attentions from the last layer
    avg_attentions = torch.mean(last_layer_attentions, dim=1)  # Average of attention weights across all heads
    normalized_attentions = avg_attentions[0].cpu().detach().numpy() / avg_attentions[0].cpu().detach().numpy().max()
    return normalized_attentions

def highlight_text(tokenizer, input_text, input_ids, attention_weights):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    highlighted_text = ""
    for token, weight in zip(tokens, attention_weights):
        token = token.replace('▁', '')  # Remove special character for readability
        color = f"rgba(255, 0, 0, {weight[0]})"  # Red color with transparency based on attention weight
        highlighted_text += f"<span style='background-color: {color}'>{token}</span> "
    return highlighted_text

def step_visualization(step_number, step_description, explanation):
    with st.spinner(f"Step {step_number}: {step_description}..."):
        time.sleep(1)
    st.success(f"Step {step_number}: {step_description} completed.")
    st.info(explanation)

# Cache functions
@st.cache_resource
def load_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name, output_attentions=True, return_dict=True)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    log_model_usage(model_name)
    return model, tokenizer

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    log_decision("Loaded embedding model 'all-MiniLM-L6-v2'")
    return model

@st.cache_resource
def load_crop_data():
    with open('crop_data.json', 'r') as file:
        data = json.load(file)
    log_decision(f"Loaded crop data from crop_data.json")
    return data

@st.cache_data
def generate_and_cache_embeddings(_embedding_model, data, _generate_context_func):
    keys = list(data.keys())
    contexts = [_generate_context_func(key, data[key]) for key in keys]
    context_embeddings = _embedding_model.encode(contexts, convert_to_tensor=True)
    embeddings = {key: embedding.cpu().numpy() for key, embedding in zip(keys, context_embeddings)}
    log_decision("Generated and cached embeddings for crop data")
    return embeddings

# Manager classes
class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = load_model(model_name)

class EmbeddingManager:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.crop_data = load_crop_data()

    def generate_context(self, key, details):
        context_lines = []
        for k, v in details.items():
            if isinstance(v, list):
                v = ', '.join(map(str, v))
            elif isinstance(v, dict):
                v = self.generate_context(k, v)  # Recursively handle nested dictionaries
            context_lines.append(f"{k.replace('_', ' ').title()}: {v}")
        return '\n'.join(context_lines)

    def find_relevant_context(self, question, embeddings, data):
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        context_embeddings = [torch.tensor(embedding) for embedding in embeddings.values()]
        cosine_scores = util.pytorch_cos_sim(question_embedding, torch.stack(context_embeddings))
        best_match_index = torch.argmax(cosine_scores).item()
        best_match_key = list(embeddings.keys())[best_match_index]
        return best_match_key, data[best_match_key], cosine_scores

class TemplateManager:
    def __init__(self, template_file='templates.json'):
        self.template_file = template_file
        self.templates = self.load_templates()

    @log_performance
    def load_templates(self):
        if os.path.exists(self.template_file):
            with open(self.template_file, 'r') as file:
                return json.load(file)
        else:
            return self.default_templates()

    def save_templates(self):
        with open(self.template_file, 'w') as file:
            json.dump(self.templates, file, indent=4)
        log_decision(f"Saved templates to {self.template_file}")

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
            # Additional templates as in original code
        }

    def get_templates(self):
        return self.templates

    def update_template(self, question_type, template, keywords):
        self.templates[question_type]["template"] = template
        self.templates[question_type]["keywords"] = [keyword.strip() for keyword in keywords.split(',')]
        self.save_templates()
        log_decision(f"Updated template for {question_type}")

    def determine_question_type(self, question):
        question = question.lower()
        for question_type, details in self.templates.items():
            if any(keyword in question for keyword in details.get("keywords", [])):
                return question_type
        return "Planting Guide"  # Default to planting guide if no keywords match

class CropGuideGenerator:
    def __init__(self, model_manager, embedding_manager, template_manager):
        self.model_manager = model_manager
        self.embedding_manager = embedding_manager
        self.template_manager = template_manager

    @log_performance
    def generate_text(self, task_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping, use_template, question_type):
        try:
            model, tokenizer = self.model_manager.model, self.model_manager.tokenizer
            input_text = self.create_input_text(task_type, question, context, use_template, question_type)
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            memory_before = memory_usage()
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                output_attentions=True,
                return_dict_in_generate=True
            )
            encoder_outputs = model.get_encoder()(inputs, output_attentions=True)
            decoder_input_ids = model._shift_right(inputs)
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=inputs.ne(tokenizer.pad_token_id),
                output_attentions=True,
                return_dict=True
            )

            attentions = decoder_outputs.attentions

            memory_after = memory_usage()
            memory_footprint = memory_after - memory_before
            answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            log_generation_details({
                "task_type": task_type,
                "question": question,
                "context": context,
                "generated_text": answer,
                "memory_footprint": memory_footprint
            })

            return format_output(answer), memory_footprint, attentions, input_text, inputs

        except Exception as e:
            st.error(f"An error occurred during text generation: {e}")
            return "", 0, None, None, None

    def create_input_text(self, task_type, question, context, use_template, question_type):
        templates = self.template_manager.get_templates()
        input_text = templates[question_type]["template"].format(question=question, context=context) if use_template else f"{context} {question}"
        if task_type == "Paraphrasing":
            input_text = f"paraphrase: {input_text}"
        elif task_type == "Summarization":
            input_text = f"summarize: {input_text}"
        elif task_type == "Question Answering":
            input_text = f"question: {question} context: {context}"
        elif task_type == "NER":
            input_text = f"ner: {input_text}"
        return input_text

# Streamlit UI
st.title("Educational Crop Growing Guide Generator")

# Initialize managers
model_manager = ModelManager("google/flan-t5-base")
embedding_manager = EmbeddingManager()
template_manager = TemplateManager()

# Load embeddings
if "embeddings" not in st.session_state:
    st.session_state.embeddings = generate_and_cache_embeddings(embedding_manager.embedding_model, embedding_manager.crop_data, embedding_manager.generate_context)

# Sidebar configuration
st.sidebar.title("Configuration")
task_type = st.sidebar.selectbox("Select Task", ["Text Generation", "Summarization", "Question Answering", "Paraphrasing", "NER"])
use_template = st.sidebar.checkbox("Use Template", value=True)
st.sidebar.title("Model Parameters")
max_length = st.sidebar.slider("Max Length", 50, 500, 300)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5)
no_repeat_ngram_size = st.sidebar.slider("No Repeat N-Gram Size", 1, 10, 2)
early_stopping = st.sidebar.checkbox("Early Stopping", value=True)
st.sidebar.title("Template Configuration")
selected_question_type = st.sidebar.selectbox("Select Question Type", list(template_manager.get_templates().keys()))
template_input = st.sidebar.text_area("Template", value=template_manager.get_templates()[selected_question_type]["template"])
keywords_input = st.sidebar.text_area("Keywords (comma separated)", value=", ".join(template_manager.get_templates()[selected_question_type]["keywords"]))
if st.sidebar.button("Save Template"):
    template_manager.update_template(selected_question_type, template_input, keywords_input)
    st.sidebar.success("Template saved successfully!")
st.sidebar.title("Cache Management")
if st.sidebar.button("Clear Cache and Reload Data"):
    st.experimental_rerun()
if st.sidebar.button("Clear Cache and Reload Templates"):
    template_manager.load_templates()
    st.experimental_rerun()
# Main input and processing section
question = st.text_input("Question", value="How to grow tomatoes?", key="question", help="Enter your question about crop growing here.")
log_question(question)

if st.button("Generate"):
    if question:
        check_memory_usage()  # Check memory usage before processing each request
        step_visualization(1, "Loading the model", "We load a pre-trained T5 model which is designed for text generation tasks. This model is capable of generating coherent and contextually appropriate text based on the given input.")
        step_visualization(2, "Loading the templates", "Templates provide a structured format for generating specific types of responses. They help in guiding the model to produce more relevant and context-specific outputs.")
        step_visualization(3, "Loading the crop data and constructing embeddings based on the model", "We use embeddings to convert textual data into numerical vectors. These vectors help in finding similarities between the question and the data. SentenceTransformers model 'all-MiniLM-L6-v2' is used for generating these embeddings.")
        
        step_visualization(4, "Getting user input", "The user input is the question asked by the student. This question will be processed to find the most relevant context from the crop data.")
        step_visualization(5, "Finding relevant context and calculating cosine similarity", "We use cosine similarity to measure the similarity between the question and each entry in the crop data. This helps in retrieving the most relevant context for the given question.")
        
        memory_before_retrieval = get_memory_usage()
        best_match_key, relevant_context, cosine_scores = embedding_manager.find_relevant_context(question, st.session_state.embeddings, embedding_manager.crop_data)
        memory_after_retrieval = get_memory_usage()
        retrieval_memory_footprint = memory_after_retrieval - memory_before_retrieval

        context = embedding_manager.generate_context("Crop", relevant_context)

        st.subheader("Detected Question Type")
        question_type = template_manager.determine_question_type(question)
        st.write(f"**{question_type}**")

        st.subheader("Context")
        st.markdown(f"```{context}```")

        # Visualization of Cosine Similarity Scores
        fig, ax = plt.subplots()
        ax.bar(embedding_manager.crop_data.keys(), cosine_scores.cpu().numpy().flatten())
        ax.set_ylabel('Cosine Similarity Score')
        ax.set_xlabel('Crop Data Entry')
        ax.set_title('Cosine Similarity Scores with Crop Data Entries')
        st.pyplot(fig)


        crop_names = list(embedding_manager.crop_data.keys())
        cosine_scores = np.random.rand(len(crop_names))  # Replace with your actual cosine similarity scores

        # Create a Plotly bar chart
        fig = go.Figure(data=[go.Bar(
            x=crop_names,
            y=cosine_scores,
            marker_color='skyblue',  # Change color if desired
        )])

        # Customize layout
        fig.update_layout(
            title='Cosine Similarity Scores with Crop Data Entries',
            xaxis_title='Crop Data Entry',
            yaxis_title='Cosine Similarity Score',
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            bargap=0.1,  # Gap between bars
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

        step_visualization(6, "Determining question type", "Based on the keywords in the question, we determine the type of question (e.g., planting guide, common issues). This helps in selecting the appropriate template for generating the response.")
        
        crop_guide_generator = CropGuideGenerator(model_manager, embedding_manager, template_manager)
        with st.spinner("Generating..."):
            guide, generation_memory_footprint, attentions, input_text, input_ids = crop_guide_generator.generate_text(
                task_type, question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping, use_template, question_type
            )

        step_visualization(7, "Generating the text", "The text generation model uses the context and the question to generate a detailed guide. Parameters like max_length, num_beams, and no_repeat_ngram_size help in controlling the quality and length of the generated text.")
        
        st.subheader("Generated Guide")
        st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{guide}</div>", unsafe_allow_html=True)

        if attentions is not None:
            normalized_attentions = normalize_attention_weights(attentions)
            highlighted_text = highlight_text(model_manager.tokenizer, input_text, input_ids, normalized_attentions)
            st.subheader("Highlighted Input Text Based on Attention Weights")
            st.markdown(f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{highlighted_text}</div>", unsafe_allow_html=True)

        step_visualization(8, "Displaying the output", "The final output is displayed to the user. This output is the detailed guide generated by the model based on the user's question and the retrieved context.")
        
        st.subheader("Memory Footprint Details")
        st.write(f"Memory Footprint for Retrieval: {retrieval_memory_footprint:.2f} MB")
        st.write(f"Memory Footprint for Generation: {generation_memory_footprint:.2f} MB")
        st.write(f"Total Memory Footprint: {retrieval_memory_footprint + generation_memory_footprint:.2f} MB")
        
        # Stop the app execution here
        st.stop()
