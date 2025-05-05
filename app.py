

import streamlit as st
import pandas as pd
import google.generativeai as genai
import io
import os
from google import genai


client = genai.Client(api_key=os.environ.get("API_KEY"))

st.title("📊 Excel Q&A Assistant")

# classifier = st.selectbox("Select a classifier", ["Classifier 1", "Classifier 2"])  # Replace with your actual classifiers
# st.write("You selected:", classifier)
classifier = ["אוכל", "תכשיטים", "אחר", "נסיעות" ,"דלק"]
dictionary = {
    "אוכל": 0,
    "תכשיטים": 0,
    "אחר": 0,
    "נסיעות": 0,
    "דלק": 0
}

# File uploader xlsx or csv
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

@st.cache_data
def get_classifier(str):
        
     # # Send the row to Gemini API for processing
        prompt = f"Given a sentence, you need to classify it to one of the following categories: {classifier}\n\nSentence: {str}"
        response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
        )
        answer = response.text
        return answer


# If a file is uploaded
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
       df = pd.read_csv(uploaded_file, encoding='windows-1255', skiprows=1)

    else:
        df = pd.read_excel(uploaded_file, skiprows=3)

    df = df.head(5)
    st.success("File uploaded and read successfully!")
    st.write("Here's a preview:")
    st.dataframe(df.head())

    df["classification"] = df.apply(lambda row: get_classifier(row.to_string(index=False)), axis=1)
    st.write("Here's the classification:")

    st.dataframe(df.head())