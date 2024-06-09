import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate responses using GPT-2
def generate_response(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Streamlit app layout
st.title("Chat with GPT-2")
st.write("This is a simple chat application using the GPT-2 model.")

# Default test prompt
default_prompt = "How to grow tomatoes?"

# Text input for user prompt
user_input = st.text_input("You:", default_prompt)

# Generate and display response when the user inputs text
if user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
    st.text_area("GPT-2:", response, height=200)
