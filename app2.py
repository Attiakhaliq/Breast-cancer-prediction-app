import csv

import streamlit as st
import pandas as pd
import joblib
import os



# Load the trained model (relative path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model2.pkl')
SAMPLE_PATH = os.path.join(BASE_DIR, 'sample_patients.csv')

try:
    model2 = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()



# ------------------------------
# App title and description
st.title("Breast Cancer Prediction App")
st.write(
    "Upload a CSV file containing cell measurement data to predict whether "
    "each tumor sample is malignant or benign."
)

# ------------------------------


st.write ("Don't have a CSV file? Download a sample to try the app:")
try:
    with open(SAMPLE_PATH, 'rb') as f:
        st.download_button(
            label="Download Sample CSV",
            data=f,
            file_name='sample_patients.csv',
            mime='text/csv'
        )
except FileNotFoundError:
    st.error("Sample CSV file not found.")




# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

required_columns = list(model2.feature_names_in_)

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)

        missing_cols = set(required_columns) - set(input_data.columns)
        extra_cols = set(input_data.columns) - set(required_columns)

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if extra_cols:
                st.warning(f"Extra columns found (will be ignored): {extra_cols}")

            # Reorder columns to match training order
            input_data = input_data[required_columns]

            st.subheader("Uploaded Data Preview")
            st.dataframe(input_data.head())

            # Predict button
            if st.button("Predict"):
                predictions = model2.predict(input_data)
                probabilities = model2.predict_proba(input_data)

                label_map = {0: 'Malignant', 1: 'Benign'}
                results = input_data.copy()
                results['Prediction'] = [label_map[p] for p in predictions]
                results['Confidence'] = probabilities.max(axis=1)

                st.subheader("Prediction Results")
                st.dataframe(results[['Prediction', 'Confidence']])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to get predictions.")