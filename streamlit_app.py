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
    gender = st.selectbox("Gender", options=["male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", options=[
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
    ])
    lunch = st.selectbox("Lunch", options=["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", options=["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    submit_button = st.form_submit_button("Predict")

if submit_button:
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
