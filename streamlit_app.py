import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit app
st.title("Student Performance Predictor")

# Form for input
with st.form("prediction_form"):
    st.header("Enter student details")
    
    gender = st.selectbox("Gender", options=["Select Gender", "male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", options=["Select Race/Ethnicity", "group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", options=[
        "Select Parental Level of Education", "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
    ])
    lunch = st.selectbox("Lunch", options=["Select Lunch", "standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", options=["Select Test Preparation Course", "none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Check if any placeholder options are still selected
    if (gender == "Select Gender" or race_ethnicity == "Select Race/Ethnicity" or
        parental_level_of_education == "Select Parental Level of Education" or
        lunch == "Select Lunch" or test_preparation_course == "Select Test Preparation Course"):
        st.error("Please fill in all the fields.")
    else:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()

        st.write("Input DataFrame:")
        st.write(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        st.write("Prediction Results:")
        st.write(results[0])
