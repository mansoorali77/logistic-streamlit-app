import streamlit as st
import numpy as np
import pickle

# Load the trained logistic regression model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger details below to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 500, 50)
sex = st.radio("Sex", ['male', 'female'])
embarked = st.radio("Port of Embarkation", ['Q', 'S'])  # Only Q and S were one-hot encoded

# Encode categorical variables
sex = 1 if sex == 'male' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Final feature vector (6 features total, in this exact order)
features = np.array([[pclass, age, fare, sex, embarked_q, embarked_s]])

# Prediction
if st.button("Predict"):
    pred = model.predict(features)
    if pred[0] == 1:
        st.success("✅ The passenger would have survived!")
    else:
        st.error("❌ The passenger would not have survived.")
