
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
import pandas as pd

# Load your CSV or DataFrame
df = pd.read_csv("compressed_data.csv.gz")

# Optional: Drop rows with missing values
df.dropna(inplace=True)

# Clean the text (you can reuse your clean_text function)
def clean_text(text):
    text = pd.Series(text).str.lower()
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n", '', regex=True)
    text = text.str.replace('\d', '', regex=True)
    text = text.str.replace(r'\[.*?\]', '', regex=True)
    text = text.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
    text = text.str.replace(r'<.*?>+', '', regex=True)
    text = text.str.replace(r'\w*\d\w*', '', regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    return text.iloc[0]

df['cleaned_text'] = df['text'].apply(clean_text)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# Train vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])

# Train classifier
model = LogisticRegression()
model.fit(X, y)
# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoder if you want to decode predictions
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

label = label_encoder.inverse_transform([prediction_index])[0]
