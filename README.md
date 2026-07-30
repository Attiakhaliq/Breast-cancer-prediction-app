# Breast Cancer Prediction App

A machine learning web app that predicts whether a breast tumor is malignant 
or benign, based on cell nuclei measurements from biopsy samples. Built as 
part of an ML coursework assignment.

## Problem Statement
Early and accurate diagnosis of breast tumors is critical for effective 
treatment. This app uses a trained classification model to assist in 
distinguishing malignant (cancerous) from benign (non-cancerous) tumors 
based on diagnostic cell measurements, serving as a decision-support tool.

## Live App
https://breast-cancer-prediction-app-t77edjezhrnptb8p9kv99c.streamlit.app/

## Dataset
Breast Cancer Wisconsin (Diagnostic) Dataset — a well-established medical 
dataset built from digitized images of fine needle aspirate (FNA) biopsies 
of breast masses. Available directly through scikit-learn's built-in datasets.

**Features used:** 30 numeric features describing cell nuclei characteristics, 
computed as mean, standard error, and "worst" (largest) values for 10 base 
measurements including:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

**Target:** Diagnosis (Malignant or Benign)

## Model
- **Algorithm:** Logistic Regression
- **Preprocessing:** Mean imputation for missing values, feature scaling 
  (StandardScaler)
- **Pipeline:** scikit-learn Pipeline combining preprocessing and model into 
  a single deployable object


## How It Works
1. User uploads a CSV file containing cell measurement data for one or more 
   tumor samples (or downloads a provided sample file to test the app)
2. The app validates that all 30 required columns are present and correctly 
   named
3. The trained model predicts malignant/benign for each sample
4. Results are displayed with prediction and confidence score

## Built With
- Python
- scikit-learn (Logistic Regression)
- Streamlit
- pandas / numpy / joblib

## Author
Attia Khaliq
