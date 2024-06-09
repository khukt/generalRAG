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
def generate_paragraph(question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping):
    input_text = (
        f"Please provide a detailed, step-by-step guide on how to grow the specified crop based on the following question and context.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Steps:"
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_beams=num_beams, 
        no_repeat_ngram_size=no_repeat_ngram_size, 
        early_stopping=early_stopping
    )
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
st.title("Crop Growing Guide Generator")
st.write("Select a crop and enter your question and context to generate a detailed guide.")

crop_choice = st.selectbox("Select Crop", ["Tomato", "Corn"])

if crop_choice == "Tomato":
    context = """
        Crop Name: Tomato
        Planting Season: Spring
        Harvest Time: Summer
        Soil Type: Well-drained, fertile soil
        Watering Needs: Regular watering, keep soil moist but not waterlogged
        Pests and Diseases: Aphids, Blight, Tomato Hornworm
    """
else:
    context = """
        Crop Name: Corn
        Planting Season: Late Spring
        Harvest Time: Late Summer to Early Fall
        Soil Type: Well-drained, loamy soil
        Watering Needs: Moderate watering, keep soil moist especially during tasseling and ear development
        Pests and Diseases: Corn Earworm, Rootworm, Corn Smut
    """

question = st.text_input("Question", value=f"How to grow {crop_choice.lower()}?")
context = st.text_area("Context", value=context)

# Additional controls for model.generate parameters
max_length = st.slider("Max Length", 50, 500, 300)
num_beams = st.slider("Number of Beams", 1, 10, 5)
no_repeat_ngram_size = st.slider("No Repeat N-Gram Size", 1, 10, 2)
early_stopping = st.checkbox("Early Stopping", value=True)

if st.button("Generate Guide"):
    with st.spinner("Generating..."):
        guide = generate_paragraph(question, context, max_length, num_beams, no_repeat_ngram_size, early_stopping)
    st.subheader("Generated Guide")
    st.write(guide)

# Cache resource decorator for efficient reloading
@st.cache_resource
def get_crop_details(crop_name):
    if crop_name == "Tomato":
        return {
            'name': 'Tomato',
            'planting_season': 'Spring',
            'harvest_time': 'Summer',
            'soil_type': 'Well-drained, fertile soil',
            'watering_needs': 'Regular watering, keep soil moist but not waterlogged',
            'pests_diseases': ['Aphids', 'Blight', 'Tomato Hornworm']
        }
    else:
        return {
            'name': 'Corn',
            'planting_season': 'Late Spring',
            'harvest_time': 'Late Summer to Early Fall',
            'soil_type': 'Well-drained, loamy soil',
            'watering_needs': 'Moderate watering, keep soil moist especially during tasseling and ear development',
            'pests_diseases': ['Corn Earworm', 'Rootworm', 'Corn Smut']
        }

crop_details = get_crop_details(crop_choice)
crop_text = (
    f"Crop Name: {crop_details['name']}\n"
    f"Planting Season: {crop_details['planting_season']}\n"
    f"Harvest Time: {crop_details['harvest_time']}\n"
    f"Soil Type: {crop_details['soil_type']}\n"
    f"Watering Needs: {crop_details['watering_needs']}\n"
    f"Pests and Diseases: {', '.join(crop_details['pests_diseases'])}\n"
)
st.write(crop_text)
