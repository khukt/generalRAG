import streamlit as st
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util

# Load and parse JSON database
@st.cache_resource
def load_json():
    data = {
        "crops": [
            {
                "id": 1,
                "name": "Tomato",
                "planting_season": "Spring",
                "harvest_time": "Summer",
                "soil_type": "Well-drained, fertile soil",
                "watering_needs": "Regular watering, keep soil moist but not waterlogged",
                "pests_diseases": ["Aphids", "Blight", "Tomato Hornworm"]
            }
        ],
        "pests_diseases": [
            {
                "id": 1,
                "name": "Aphids",
                "affected_crops": ["Tomato", "Cucumber"],
                "symptoms": "Yellowing leaves, sticky residue",
                "treatment": "Neem oil, ladybugs"
            }
        ],
        "weather": [
            {
                "region": "Midwest",
                "climate": "Temperate",
                "average_temp": "15°C",
                "rainfall": "30 inches",
                "best_crops": ["Corn", "Soybean"]
            }
        ],
        "market": [
            {
                "id": 1,
                "crop": "Tomato",
                "current_price": "$2.5 per kg",
                "demand_trends": "High in summer"
            }
        ]
    }
    return data

data = load_json()

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_models():
    model_name = "google/flan-t5-base"
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return t5_model, t5_tokenizer, sbert_model

t5_model, t5_tokenizer, sbert_model = load_models()

# Function to generate text based on input question and context
def generate_paragraph(question, context):
    input_text = (
        f"Please provide a detailed, step-by-step guide on how to grow tomatoes based on the following question and context.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Steps:"
    )
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_output(answer)

# Function to format the output into a well-written paragraph
def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences if sentence)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    return formatted_output

# Function to find the most relevant context from the JSON database using cosine similarity
def get_relevant_context(question, data, sbert_model):
    contexts = []
    for crop in data['crops']:
        context = f"Crop Name: {crop['name']}, Planting Season: {crop['planting_season']}, Harvest Time: {crop['harvest_time']}, Soil Type: {crop['soil_type']}, Watering Needs: {crop['watering_needs']}, Pests and Diseases: {', '.join(crop['pests_diseases'])}"
        contexts.append(context)
    for pest in data['pests_diseases']:
        context = f"Pest Name: {pest['name']}, Affected Crops: {', '.join(pest['affected_crops'])}, Symptoms: {pest['symptoms']}, Treatment: {pest['treatment']}"
        contexts.append(context)
    for weather in data['weather']:
        context = f"Region: {weather['region']}, Climate: {weather['climate']}, Average Temperature: {weather['average_temp']}, Rainfall: {weather['rainfall']}, Best Crops: {', '.join(weather['best_crops'])}"
        contexts.append(context)
    for market in data['market']:
        context = f"Crop: {market['crop']}, Current Price: {market['current_price']}, Demand Trends: {market['demand_trends']}"
        contexts.append(context)

    question_embedding = sbert_model.encode(question, convert_to_tensor=True)
    context_embeddings = sbert_model.encode(contexts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(question_embedding, context_embeddings)
    top_context_index = torch.argmax(cosine_scores).item()
    return contexts[top_context_index]

# Streamlit UI
st.title("Tomato Growing Guide Generator")
st.write("Enter your question to generate a detailed guide.")

question = st.text_input("Question", value="How to grow tomato?")

if st.button("Generate Guide"):
    with st.spinner("Generating..."):
        context = get_relevant_context(question, data, sbert_model)
        guide = generate_paragraph(question, context)
    st.subheader("Generated Guide")
    st.write(guide)
    st.subheader("Relevant Context")
    st.write(context)
