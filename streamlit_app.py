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

def context_to_sentence(context):
    # Split the context into key-value pairs
    context_dict = dict(item.split(": ") for item in context.split("\n") if ": " in item)
    
    # Create a coherent sentence from the context
    sentence = (
        f"To grow {context_dict.get('Crop Name', 'tomatoes')}, you should plant them in the {context_dict.get('Planting Season', 'spring')}."
        f" The harvest time is around {context_dict.get('Harvest Time', '75 days')}."
        f" They thrive best in {context_dict.get('Soil Type', 'loamy')} soil."
        f" Watering needs are approximately {context_dict.get('Watering Needs', '1 inch per week')}."
        f" Be aware of pests and diseases such as {context_dict.get('Pests and Diseases', 'aphids and blight')}."
    )
    return sentence

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
    context_sentence = context_to_sentence(context)
    prompt = f"Question: {question}\nContext: {context_sentence}\nExplanation:"
    explanation = generate_explanation(prompt)
    st.write(f"**Context Sentence:** {context_sentence}")
    st.write(f"**Explanation:** {explanation}")
