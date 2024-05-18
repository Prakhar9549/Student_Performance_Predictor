# streamlit_app.py
import streamlit as st
import threading
from flask import Flask, request, render_template
from werkzeug.serving import make_server
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results[0])

class FlaskThread(threading.Thread):
    def run(self):
        make_server('0.0.0.0', 5000, app).serve_forever()

# Start Flask in a separate thread
flask_thread = FlaskThread()
flask_thread.start()

# Streamlit app
st.title("Streamlit and Flask Integration")
st.write("Flask app is running on http://localhost:5000")

st.markdown("""
    ## Instructions
    The Flask app is running in the background. You can access it at [http://localhost:5000](http://localhost:5000).
""")

# Correctly import the components module
import streamlit.components.v1 as components

# Embedding an iframe to display the Flask app within the Streamlit app (optional)
components.iframe(src="http://localhost:5000", height=600, scrolling=True)
