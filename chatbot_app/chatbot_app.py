
# chatbot_app.py
import streamlit as st
import pickle
import pandas as pd # Import pandas here for clean_text

# Define the clean_text function (assuming it's defined elsewhere in the notebook)
def clean_text(text):
    # This is a placeholder. Replace with your actual clean_text implementation
    # Example:
    text = pd.Series(text).str.lower()
    text = text.str.replace(r'[^\w\s]', '', regex = True)
    text = text.str.replace("\n" , '', regex = True) # Use double backslash for newline
    text = text.str.replace('\d', '', regex = True)
    text = text.str.replace(r'\[.*?\]', '', regex = True) # Use double backslash for square brackets
    text = text.str.replace(r'https?://\S+|www\.\S+', '', regex = True) # Use double backslash for period
    text = text.str.replace(r'<.*?>+', '', regex = True)
    text = text.str.replace(r'\w*\d\w*', '', regex = True) # Use double backslash for digit
    return text.iloc[0] # Return the cleaned string

# Assuming the vectorizer and model are already trained and saved
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please train and save them first.")
    vectorizer = None
    model = None


st.title("Mental Health Detection Chatbot")
user_input = st.text_input("How are you feeling today?")

if user_input and vectorizer and model:
    cleaned = clean_text(user_input)
    vec_input = vectorizer.transform([cleaned])
    prediction_index = model.predict(vec_input)[0]

    # Define the mapping of indices to category labels
    label_mapping = {
        0: 'Anxiety',
        1: 'Normal',
        2: 'Depression',
        3: 'Suicidal',
        4: 'Stress',
        5: 'Bipolar',
        6: 'Personality disorder'
    }
    prediction_label = label_mapping.get(prediction_index, "Unknown") # Use .get for safety

    st.write(f"Prediction: **{prediction_label}**")

    if prediction_label == "Depression":
        st.warning("You might be showing signs of depression. Consider talking to a professional.")
        st.info(" 988 Lifeline's") # Consider adding a full link or contact info
