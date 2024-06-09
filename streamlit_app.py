import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_model():
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate text based on input question and context
def generate_paragraph(question, context):
    input_text = f"Generate a detailed guide based on the following question and context.\n\nQuestion: {question}\nContext: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_output(answer)

# Function to format the output into a well-written paragraph
def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences)
    return formatted_output

# The expected output for comparison
expected_output = ("Tomatoes are best planted in the spring when the soil begins to warm up. "
                   "They thrive in well-drained, fertile soil, which should be kept consistently moist but not waterlogged through regular watering. "
                   "Expect to harvest your tomatoes in the summer when they reach full maturity. "
                   "Watch out for common pests and diseases such as aphids, blight, and tomato hornworm. "
                   "Taking preventive measures and regular monitoring can help maintain healthy plants and ensure a successful harvest.")

# Streamlit UI
st.title("Tomato Growing Guide Generator")
st.write("Enter your question and context about growing tomatoes to generate a detailed guide.")

question = st.text_input("Question", value="How to grow tomato?")
context = st.text_area("Context", value="""
Crop Name: Tomato
Planting Season: Spring
Harvest Time: Summer
Soil Type: Well-drained, fertile soil
Watering Needs: Regular watering, keep soil moist but not waterlogged
Pests and Diseases: Aphids, Blight, Tomato Hornworm
""")

if st.button("Generate Guide"):
    with st.spinner("Generating..."):
        guide = generate_paragraph(question, context)
        result = guide == expected_output
    st.subheader("Generated Guide")
    st.write(guide)
    st.subheader("Does the generated guide match the expected output?")
    st.write(result)
