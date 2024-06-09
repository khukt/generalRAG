import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def generate_explanation(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
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

if st.button("Generate Explanation"):
    prompt = f"Question: {question}\nContext: {context}\nExplanation:"
    explanation = generate_explanation(prompt)
    st.write(explanation)
