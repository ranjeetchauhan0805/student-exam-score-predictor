import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "student_exam_scores.csv"

df = pd.read_csv(DATA_PATH)

st.sidebar.title("About")
st.sidebar.info(
    """
    **Student Exam Score Predictor**

    Model: Linear Regression

    Built using:
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    """
)

st.set_page_config(page_title="Student Exam Score Predictor", 
                   page_icon="🎓", 
                   layout="centered")

st.title("🎓 Student Exam Score Predictor")
st.write("Enter the student's details.", "The model will predict the expected exam score based on the input features.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "exam_score_predictor.pkl"

model = joblib.load(MODEL_PATH)

hours = st.number_input(
    "Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

sleep = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=8.0
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

if st.button("Predict Score"):

    new_student = pd.DataFrame({
        "hours_studied": [hours],
        "sleep_hours": [sleep],
        "attendance_percent": [attendance],
        "previous_scores": [previous]
    })

    prediction = model.predict(new_student)
    
    score = prediction[0]
    st.metric("Predicted Exam Score", f"{score:.2f}")

st.sidebar.subheader("Dataset")

st.sidebar.metric("Students", len(df))
st.sidebar.metric("Average Score", round(df["exam_score"].mean(), 2))
st.sidebar.metric("Highest Score", round(df["exam_score"].max(), 2))