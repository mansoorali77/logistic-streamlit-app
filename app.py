import streamlit as st
import numpy as np
import pickle

# Load the trained logistic regression model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the details below to predict survival")

# Input fields (based on model: ['const', 'Pclass', 'Age', 'Fare'])
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 500, 50)

# Add constant manually for 'const' column
const = 1.0

# Input vector with exact order
features = np.array([[const, pclass, age, fare]])

# Prediction
if st.button("Predict"):
    pred = model.predict(features)
    if pred[0] == 1:
        st.success("✅ The passenger would have survived!")
    else:
        st.error("❌ The passenger would not have survived.")
