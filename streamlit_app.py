import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Cache the model and tokenizer to optimize memory usage
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate text based on input question and context
def generate_paragraph(question, context):
    input_text = f"Generate a detailed guide for growing tomatoes based on the following question and context.\n\nQuestion: {question}\n\nContext: {context}\n\nGuide:"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=300, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_output(answer)

# Function to format the output into a well-written paragraph
def format_output(output):
    sentences = output.split('. ')
    formatted_output = '. '.join(sentence.capitalize() for sentence in sentences if sentence)
    if not formatted_output.endswith('.'):
        formatted_output += '.'
    return formatted_output

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
    st.subheader("Generated Guide")
    st.write(guide)

# Cache resource decorator for efficient reloading
@st.cache_resource
def get_crop_details():
    crop = {
        'name': 'Tomato',
        'planting_season': 'Spring',
        'harvest_time': 'Summer',
        'soil_type': 'Well-drained, fertile soil',
        'watering_needs': 'Regular watering, keep soil moist but not waterlogged',
        'pests_diseases': ['Aphids', 'Blight', 'Tomato Hornworm']
    }
    return crop

crop = get_crop_details()
crop_text = (
    f"Crop Name: {crop['name']}\n"
    f"Planting Season: {crop['planting_season']}\n"
    f"Harvest Time: {crop['harvest_time']}\n"
    f"Soil Type: {crop['soil_type']}\n"
    f"Watering Needs: {crop['watering_needs']}\n"
    f"Pests and Diseases: {', '.join(crop['pests_diseases'])}\n"
)
st.write(crop_text)
