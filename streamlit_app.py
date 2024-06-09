import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    # Add a padding token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def generate_explanation(prompt, max_length, temperature, top_k, top_p, repetition_penalty):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)["attention_mask"]
    
    outputs = model.generate(
        inputs,
        max_length=150,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id
    )
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation

st.title("Tomato Growing Guide")

st.write("Enter your question and context to get an explanation on how to grow tomatoes.")

# Input fields
question = st.text_input("Your question:", "How to grow tomatoes?")
context = st.text_area("Context provided to model:", 
                       "Crop Name: Tomato\n"
                       "Planting Season: Spring\n"
                       "Harvest Time: 75 days\n"
                       "Soil Type: Loamy\n"
                       "Watering Needs: 1 inch per week\n"
                       "Pests and Diseases: Aphids, Blight")

# Control variables
max_length = st.slider("Max Length", 50, 300, 150)
temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
top_k = st.slider("Top K", 0, 100, 50)
top_p = st.slider("Top P", 0.0, 1.0, 0.9)
repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2)

if st.button("Generate Explanation"):
    prompt = f"Question: {question}\nContext: {context}\nExplanation:"
    explanation = generate_explanation(prompt, max_length, temperature, top_k, top_p, repetition_penalty)
    st.write(explanation)
