import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")

st.title("📊 Student Performance Predictor")

# Inputs
study_hours = st.slider("Study Hours", 1, 10, 5)
attendance = st.slider("Attendance (%)", 50, 100, 75)
parent_education = st.slider("Parent Education Level (1-5)", 1, 5, 3)
sleep_hours = st.slider("Sleep Hours", 4, 10, 7)

input_data = np.array([[study_hours, attendance, parent_education, sleep_hours]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Final Grade: {prediction:.2f}")

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    st.subheader("Feature Importance")

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)