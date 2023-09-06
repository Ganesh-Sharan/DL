import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = keras.models.load_model("model.h5")

# Function to preprocess input data
def preprocess_input(data):
    cols = ['gender',  'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    label_encoder = LabelEncoder()
    data[cols] = data[cols].apply(label_encoder.fit_transform)
    return data

# Map class labels to text labels
class_labels = {0: "NoStroke", 1: "Strokes"}

# Streamlit app
def main():
    st.title("Health Prediction App")
    st.sidebar.header("Input Features")

    # Create input fields for user input
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 0, 100, 25)
    hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.sidebar.slider("BMI", 0.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["Unknown", "Never_smoked", "formerly_smoked", "smokes"])

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Preprocess the input data
    input_data = preprocess_input(input_data)

    if st.button("Predict"):
        # Make predictions
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        
        # Map the predicted class label to text label
        predicted_label = class_labels.get(predicted_class, "Unknown")
        
        st.write(f"Predicted Class: {predicted_label}")

if __name__ == "__main__":
    main()
