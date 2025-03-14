import streamlit as st
import joblib
import numpy as np
import pandas as pd

#  Load the trained model, scaler, and encoders
model = joblib.load("linear_regression.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # If categorical encoding was used

# Streamlit UI
st.title("IBD Prediction App (Linear Regression)")

st.write("Enter patient details to predict IBD outcome:")

# Create dynamic input fields based on dataset
feature_inputs = {}
for col in ["Feature1", "Feature2"]:  # Replace with actual feature names from your dataset
    feature_inputs[col] = st.number_input(f"Enter {col}", min_value=0, max_value=100, value=50)

# Make Prediction
if st.button("Predict"):
    # Convert user input to a DataFrame
    input_df = pd.DataFrame([feature_inputs])

    # Encode categorical features if necessary
    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = le.transform(input_df[col])

    # Standardize input features
    input_scaled = scaler.transform(input_df)

    # Predict using the model
    prediction = model.predict(input_scaled)
    
    # Show result
    st.write(f" Predicted IBD Type: {prediction[0]:.2f}")
