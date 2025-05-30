# -*- coding: utf-8 -*-

"""mental Health

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1u-pOql7edDjJ9JXqk4QJceWxsaKMjTIi
"""
import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import xgboost as xgb
import lightgbm as lgb
import streamlit as st

import os
os.system("pip install numpy pandas scikit-learn nltk transformers torch streamlit")

import zipfile
import pandas as pd

with zipfile.ZipFile("data.zip") as z:
    with z.open("Combined Data.csv") as f:
        df = pd.read_csv(f)

df.head()

df.info()

df['statement'] = df['statement'].fillna('')

df.head()

df.drop(["Unnamed: 0"], axis = 1, inplace = True)

df["status"].unique()

df[df["status"] == "Anxiety"]["statement"][1]

df["status"] = df["status"].map({'Anxiety':0, 'Normal':1, 'Depression':2, 'Suicidal':3, 'Stress':4, "Bipolar": 5, "Personality disorder": 6})

def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    text = text.str.replace(r'[^\w\s]', '', regex = True)
    text = text.str.replace("\n" , '', regex = True)
    text = text.str.replace('\d', '', regex = True)
    text = text.str.replace(r'\[.*?\]', '', regex = True)
    text = text.str.replace(r'https?://\S+|www\.\S+', '', regex = True)
    text = text.str.replace(r'<.*?>+', '', regex = True)
    text = text.str.replace(r'\w*\d\w*', '', regex = True)
    return text

df["statement"] = clean_text(df["statement"])

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def remove_stopwords(text):
    stop_words = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["statement"] = remove_stopwords(df["statement"])

delete = pd.Series(' '.join(df['statement']).split()).value_counts()[-1000:]
df['statement'] = df['statement'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))

import spacy

nlp = spacy.load('en_core_web_sm')

def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    return " ".join([token.lemma_ for token in doc])

df['statement'] = df['statement'].apply(lemmatize_sentence)

df.head()

df.isnull().sum()

y = df['status']
X = df.drop('status', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train

def plot_wordcloud(text, title=None, save_path=None):
    wordcloud = WordCloud(width=800, height=400, colormap = 'BuPu_r',
                          background_color='white',
                          contour_width=3, contour_color='black').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=20)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(text, title=None, save_path=None):
    wordcloud = WordCloud(width=800, height=400, colormap = 'BuPu_r',
                          background_color='white',
                          contour_width=3, contour_color='black').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=20)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

all_text = ' '.join(X_train['statement'])
plot_wordcloud(all_text)

y.unique().tolist()

def count_vec_boost(model, text_col, max_df, max_features, ngram_range, n_estimators, max_depth=3, learning_rate=0.03, verbose = False):
    """
    Vectorizes text data using CountVectorizer, trains a classifier model, and evaluates its performance.

    Parameters:
    - model: Classifier model (e.g., XGBoostClassifier, LightGBMClassifier, CatBoostClassifier)
    - text_col: Name of the text column in the dataset
    - max_df: Maximum document frequency threshold for CountVectorizer
    - max_features: Maximum number of features for CountVectorizer
    - ngram_range: Tuple specifying the range of n-grams (e.g., (1, 2) for unigrams and bigrams)
    - stop_words: List of stop words to be removed during vectorization
    - n_estimators: Number of estimators for the classifier model
    - max_depth: Maximum depth of the decision trees (if applicable)
    - learning_rate: Learning rate for gradient boosting classifiers (if applicable)

    Returns:
    - None

    Prints:
    - Classification report showing precision, recall, and F1-score for each class.
    - Confusion matrix visualizing predicted vs. true labels.

    Example usage:
    ```
    vec_pred(RandomForestClassifier, 'text_column', max_df=0.8, max_features=1000, ngram_range=(1, 2),
             stop_words=['english'], n_estimators=100, max_depth=10)
    ```

    """
    best_vectorizer = CountVectorizer(max_df=max_df, max_features=max_features, ngram_range=ngram_range)
    X_train_vector = best_vectorizer.fit_transform(X_train[text_col])
    best_classifier = model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    best_classifier.fit(X_train_vector, y_train)
    X_test_vector = best_vectorizer.transform(X_test[text_col])
    y_pred = best_classifier.predict(X_test_vector)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    class_labels = y.unique().tolist()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def tfidf_vec_boost(model, text_col, max_df, max_features, ngram_range, n_estimators, max_depth=3, learning_rate=0.03, verbose = False):
    """
    Vectorizes text data using TfidfVectorizer, trains a classifier model, and evaluates its performance.

    Parameters:
    - model: Classifier model (e.g., XGBoostClassifier, LightGBMClassifier, CatBoostClassifier)
    - text_col: Name of the text column in the dataset
    - max_df: Maximum document frequency threshold for TfidfVectorizer
    - max_features: Maximum number of features for TfidfVectorizer
    - ngram_range: Tuple specifying the range of n-grams (e.g., (1, 2) for unigrams and bigrams)
    - stop_words: List of stop words to be removed during vectorization
    - n_estimators: Number of estimators for the classifier model
    - max_depth: Maximum depth of the decision trees (if applicable)
    - learning_rate: Learning rate for gradient boosting classifiers (if applicable)

    Returns:
    - None

    Prints:
    - Classification report showing precision, recall, and F1-score for each class.
    - Confusion matrix visualizing predicted vs. true labels.

    Example usage:
    ```
    vec_boost_tfidf(RandomForestClassifier, 'text_column', max_df=0.8, max_features=1000, ngram_range=(1, 2),
                    stop_words=['english'], n_estimators=100, max_depth=10)
    ```
    """
    # Vectorize text data using TfidfVectorizer
    best_vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features, ngram_range=ngram_range)
    X_train_vector = best_vectorizer.fit_transform(X_train[text_col])

    # Initialize and train the classifier model
    best_classifier = model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, verbose = verbose)
    best_classifier.fit(X_train_vector, y_train)

    # Transform test data using the trained vectorizer
    X_test_vector = best_vectorizer.transform(X_test[text_col])

    # Predict test labels
    y_pred = best_classifier.predict(X_test_vector)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_labels = y_test.unique().tolist()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def pipe_boosting(vectorizer, classifier, X_grid, y_grid):
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])

    hyperparameters = {
        'vectorizer__ngram_range': [(1, 3)],
        'vectorizer__max_df': [1.0],
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]}

    grid_search = GridSearchCV(pipeline, hyperparameters, cv=5, n_jobs=-1, verbose=1)

    grid_search.fit(X_grid, y_grid)

    print("Best hyperparameters:", grid_search.best_params_)

def predict_new_text(text, c_vectorizer, model):
    """
    Predicts the category of a given text using the trained vectorizer and model.

    Parameters:
    - text: The input text to be classified
    - c_vectorizer: The trained CountVectorizer or TfidfVectorizer
    - model: The trained classification model

    Returns:
    - dict: The predicted category label and its corresponding index
    """
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

    # Transform the input text using the vectorizer
    text_vector = c_vectorizer.transform([text])

    # Predict the category index
    predicted_index = model.predict(text_vector)[0]

    # Map the predicted index to the category label
    predicted_label = label_mapping[predicted_index]

    return {predicted_label: predicted_index}

"""%%time
pipe_boosting(vectorizer = CountVectorizer(),
classifier = XGBClassifier(verbose = -1),
X_grid = X_train['statement'],
y_grid = y_train)"""

best_hyperparameters = {'learning_rate': 0.07, 'n_estimators': 200, 'max_df': 1.0, 'max_features': 10000, 'ngram_range': (1, 3)}

from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


best_hyperparameters = {'learning_rate': 0.07, 'n_estimators': 200, 'max_df': 1.0, 'max_features': 10000, 'ngram_range': (1, 3)}

count_vec_boost(XGBClassifier, "statement", **best_hyperparameters, verbose = False)

c_vectorizer = CountVectorizer(max_df = 1.0, max_features = 2000, ngram_range = (1, 3))
X_train_vector = c_vectorizer.fit_transform(X_train['statement'])
X_test_vector = c_vectorizer.transform(X_test['statement'])

model = XGBClassifier(**best_hyperparameters)
model.fit(X_train_vector, y_train)

text = df["statement"].sample(1)
text_index = text.index
print(text.index)
text = text.values[0]
text

predict_new_text(text,c_vectorizer, model)

df.loc[text_index]["status"]

tfidf_vec_boost(XGBClassifier, "statement", 1.0, 2000, (1, 3), 200, 5, 0.07, verbose = False)

"""%%time
pipe_boosting(vectorizer = TfidfVectorizer(),
classifier = LGBMClassifier(force_col_wise=True, verbose = -1),
X_grid = X_train['statement'],
y_grid = y_train)"""

best_params = {'learning_rate': 0.06, 'n_estimators': 200, 'max_df': 1.0, 'max_features': 2000, 'ngram_range': (1, 3)}

lightgbm

from lightgbm import LGBMClassifier

tfidf_vec_boost(LGBMClassifier, "statement", **best_params, verbose = -1)

model = XGBClassifier(**best_hyperparameters, verbose = False)
model.fit(X_train_vector, y_train)

"""text = df["statement"].sample(1)
text_index = text.index
print(text.index)
text = text.values[0]
text"""

predict_new_text(text, c_vectorizer, model)

df.loc[text_index]["status"]

# chatbot_app.py
import streamlit as st

st.title("Mental Health Detection Chatbot")
user_input = st.text_input("How are you feeling today?")

if user_input:
    cleaned = clean_text(user_input)
    vec_input = vectorizer.transform([cleaned])
    prediction = model.predict(vec_input)[0]
    st.write(f"Prediction: **{prediction}**")

    if prediction == "depression":
        st.warning("You might be showing signs of depression. Consider talking to a professional.")
        st.info(" 988 Lifeline's  ")

import pickle

# After training
pickle.dump(model, open("model.pkl", "wb"))

# Assuming c_vectorizer is the CountVectorizer instance you want to save
pickle.dump(c_vectorizer, open("vectorizer.pkl", "wb")) # Changed 'vectorizer' to 'c_vectorizer'

# chatbot_app.py

import streamlit as st
import pickle

# Define the clean_text function (assuming it's defined elsewhere in the notebook)
def clean_text(text):
    # This is a placeholder. Replace with your actual clean_text implementation
    # Example:
    import pandas as pd # Import pandas if needed
    text = pd.Series(text).str.lower()
    text = text.str.replace(r'[^\w\s]', '', regex = True)
    text = text.str.replace("\n" , '', regex = True)
    text = text.str.replace('\d', '', regex = True)
    text = text.str.replace(r'\[.*?\]', '', regex = True)
    text = text.str.replace(r'https?://\S+|www\.\S+', '', regex = True)
    text = text.str.replace(r'<.*?>+', '', regex = True)
    text = text.str.replace(r'\w*\d\w*', '', regex = True)
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

from google.colab import drive
drive.mount('/content/drive')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd # Make sure pandas is imported
import numpy as np # Make sure numpy is imported if used elsewhere

# Ensure the DataFrame 'df' is loaded and processed before this cell
# (This assumes the previous cells defining and cleaning df have been run)

# Assuming 'df' and the 'status' column with numerical labels are already defined from previous cells.
# If not, you need to re-run the cells that load and preprocess 'df'.

# You must ensure the cell where df is loaded (e.g., df = pd.read_csv(...))
# was executed before this cell.

# Use the 'statement' column for text data and 'status' for labels
X = df['statement']
y = df['status'] # Use the already processed 'status' column

# **Ensure no NaN values are present in the text data before vectorization**
# Although fillna('') was used earlier, let's re-apply it to be sure.
X = X.fillna('')

# Train vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X) # Fit transform the text data

# Train classifier
model = LogisticRegression()
model.fit(X_vectorized, y) # Fit the model on the vectorized data and numerical labels

# You can save the vectorizer and model if needed later
# pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
# pickle.dump(model, open("logistic_regression_model.pkl", "wb"))

print("Model training complete.")

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

# Define prediction_index with the value you want to decode (e.g., from model.predict)
prediction_index = 0 # Replace 0 with the actual index you get from a prediction

# Get the label using the mapping
label = label_mapping.get(prediction_index, "Unknown") # Use .get for safety

print(f"The decoded label for index {prediction_index} is: {label}")
