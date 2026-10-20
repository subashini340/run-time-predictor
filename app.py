import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Marathon Predictor", page_icon="🏃‍♂️", layout="centered")

st.title("🏃‍♂️ Marathon Finish Time Predictor")
st.write("Enter your details and early race stats to predict your final finish time!")

# Load the trained model
try:
    model = joblib.load('marathon_model.pkl')
except:
    st.error("Model file not found! Please run train.py first to generate marathon_model.pkl.")

st.sidebar.header("Runner Details")

# Inputs
age = st.sidebar.slider("Age", 18, 80, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
is_female = 1 if gender == "Female" else 0

st.sidebar.header("Race Stats")
time_5k = st.sidebar.number_input("5K Split Time (in minutes)", min_value=10.0, max_value=60.0, value=25.0, step=0.5)
pace_rate = st.sidebar.number_input("Pace Rate (Fatigue factor, e.g. 0.5)", min_value=-2.0, max_value=5.0, value=0.5, step=0.1)

# Predict
if st.button("Predict Finish Time"):
    input_data = pd.DataFrame([{
        "5K_min": time_5k,
        "Age": age,
        "is_female": is_female,
        "pace_rate": pace_rate
    }])
    
    prediction = model.predict(input_data)[0]
    
    # Convert to hours and minutes
    hours = int(prediction // 60)
    minutes = int(prediction % 60)
    
    st.success(f"### Predicted Finish Time: **{hours} hours and {minutes} minutes**")
    st.balloons()