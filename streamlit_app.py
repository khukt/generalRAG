import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_explanation(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, early_stopping=True)
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
    explanation = generate_explanation(question, context)
    st.write(explanation)
