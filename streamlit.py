#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model
model_path = "best_random_forest_model.joblib"  # Update with your model's path
model = joblib.load(model_path)

# Define all the columns used during training
encoded_columns = [
    'Product', 'Region', 'Year', 'Month', 'Day', 'Feature1', 'Feature2', 
    'Product_A', 'Product_B', 'Product_C', 'Region_North', 'Region_South', 
    'Region_East', 'Region_West'
    # Add other columns as needed
]

# Title and description
st.title("Sales Prediction App")
st.write("Predict sales based on product, region, date, and other features.")

# User input widgets
product = st.selectbox("Product", ['Product_A', 'Product_B', 'Product_C'])
region = st.selectbox("Region", ['North', 'South', 'East', 'West'])
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1)
feature1 = st.number_input("Feature 1", min_value=0.0, step=0.1)
feature2 = st.number_input("Feature 2", min_value=0.0, step=0.1)

# Create a DataFrame with all possible one-hot encoded columns initialized to 0
user_input = pd.DataFrame(0, index=[0], columns=encoded_columns)

# Set the selected values to 1 if they exist in the one-hot encoded columns
if f'Product_{product}' in user_input.columns:
    user_input[f'Product_{product}'] = 1

if f'Region_{region}' in user_input.columns:
    user_input[f'Region_{region}'] = 1

# Add the date components and other features
user_input['Year'] = year
user_input['Month'] = month
user_input['Day'] = day
user_input['Feature1'] = feature1
user_input['Feature2'] = feature2

# Ensure all columns are in the correct order expected by the model
user_input = user_input.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict and display the result
if st.button("Predict"):
    prediction = model.predict(user_input)
    st.write(f"Predicted Sales: {prediction[0]:.2f}")


