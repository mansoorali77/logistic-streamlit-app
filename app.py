
import streamlit as st
import numpy as np
import pickle

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
sex = st.radio("Sex", ['male', 'female'])
embarked = st.radio("Embarked", ['Q', 'S'])

# Process inputs
sex = 1 if sex == 'male' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Feature vector [Pclass, Age, Fare, Sex, Embarked_Q, Embarked_S]
features = np.array([[pclass, age, fare, sex, embarked_q, embarked_s]])

# Predict
if st.button("Predict"):
    pred = model.predict(features)
    if pred[0] == 1:
        st.success("✅ The passenger would have survived!")
    else:
        st.error("❌ The passenger would not have survived.")
