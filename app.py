import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import streamlit as st
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Streamlit Page Config
st.set_page_config(page_title="Cardiovascular Risk Prediction", layout="wide")

# Title and Description
st.title("🩺 Cardiovascular Risk Prediction")
st.markdown("This application predicts the risk of cardiovascular disease using **Machine Learning** and **Deep Learning models.**")

# Define feature categories
numerical_features = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi"]
categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

# Dummy dataset for fitting the preprocessor
dummy_data = pd.DataFrame({
    "age": [40, 50, 60], 
    "height": [165, 170, 175], 
    "weight": [65, 75, 85], 
    "ap_hi": [120, 130, 140], 
    "ap_lo": [80, 85, 90], 
    "bmi": [24, 26, 28], 
    "gender": [1, 2, 1], 
    "cholesterol": [1, 2, 3], 
    "gluc": [1, 2, 3], 
    "smoke": [0, 1, 0], 
    "alco": [0, 1, 0], 
    "active": [1, 0, 1]
})

# Define Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# ✅ **Fix: Fit the preprocessor before using it**
preprocessor.fit(dummy_data)

# Function to preprocess user input
def preprocess_input(data):
    df = pd.DataFrame([data], columns=numerical_features + categorical_features)
    return preprocessor.transform(df)  # No more NotFittedError

# Load trained models (Check if file exists)
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path) if model_path.endswith(".pkl") else tf.keras.models.load_model(model_path)
    else:
        st.error(f"❌ Model file '{model_path}' not found. Please retrain or check file path.")
        return None

rf_model = load_model("random_forest_model.pkl")
xgb_model = load_model("xgboost_model.pkl")
dl_model = load_model("deep_learning_model.h5")

# Sidebar for user input
st.sidebar.header("🔍 Enter Patient Details")

# User Input Widgets
age = st.sidebar.slider("Age (years)", 30, 80, 50)
height = st.sidebar.number_input("Height (cm)", 120, 220, 170, step=1)
weight = st.sidebar.number_input("Weight (kg)", 40, 180, 70, step=1)
ap_hi = st.sidebar.number_input("Systolic Blood Pressure", 90, 180, 120, step=1)
ap_lo = st.sidebar.number_input("Diastolic Blood Pressure", 60, 120, 80, step=1)

# BMI Calculation (handling zero height error)
bmi = weight / ((height / 100) ** 2) if height > 0 else 0

gender = st.sidebar.selectbox("Gender (1: Female, 2: Male)", [1, 2])
cholesterol = st.sidebar.selectbox("Cholesterol Level (1: Normal, 2: Above Normal, 3: High)", [1, 2, 3])
gluc = st.sidebar.selectbox("Glucose Level (1: Normal, 2: Above Normal, 3: High)", [1, 2, 3])
smoke = st.sidebar.selectbox("Do you smoke? (0: No, 1: Yes)", [0, 1])
alco = st.sidebar.selectbox("Do you consume alcohol? (0: No, 1: Yes)", [0, 1])
active = st.sidebar.selectbox("Are you physically active? (0: No, 1: Yes)", [0, 1])

# When the user clicks the Predict button
if st.sidebar.button("🔍 Predict Risk"):
    user_data = [age, height, weight, ap_hi, ap_lo, bmi, gender, cholesterol, gluc, smoke, alco, active]
    X_input = preprocess_input(user_data)
    
    # Predictions (Handle missing models)
    rf_pred = rf_model.predict_proba(X_input)[0][1] if rf_model else None
    xgb_pred = xgb_model.predict_proba(X_input)[0][1] if xgb_model else None
    dl_pred = dl_model.predict(X_input)[0][0] if dl_model else None  # Probability score
    
    # Display Results
    st.subheader("🩸 Prediction Results:")
    
    predictions = {}
    
    if rf_pred is not None:
        predictions["Random Forest"] = rf_pred
        st.write(f"🌳 **Random Forest Prediction**: {'🔴 High Risk' if rf_pred > 0.5 else '🟢 Low Risk'} (Score: {rf_pred:.2f})")
    if xgb_pred is not None:
        predictions["XGBoost"] = xgb_pred
        st.write(f"🚀 **XGBoost Prediction**: {'🔴 High Risk' if xgb_pred > 0.5 else '🟢 Low Risk'} (Score: {xgb_pred:.2f})")
    if dl_pred is not None:
        predictions["Deep Learning"] = dl_pred
        st.write(f"🤖 **Deep Learning Prediction**: {'🔴 High Risk' if dl_pred > 0.5 else '🟢 Low Risk'} (Score: {dl_pred:.2f})")
    
    if all(model is None for model in [rf_pred, xgb_pred, dl_pred]):
        st.warning("⚠ No models were available for prediction. Please check your files.")
    
    # 📊 Graphical Representation of Predictions
    if predictions:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(predictions.keys()), list(predictions.values()), color=['green' if p < 0.5 else 'red' for p in predictions.values()])
        ax.set_xlabel("Probability of High Cardiovascular Risk")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='black', linestyle='--', label="Risk Threshold")
        ax.legend()
        st.pyplot(fig)
    
    # **Show a Final Decision**
    avg_risk = np.mean(list(predictions.values())) if predictions else 0
    st.markdown("---")
    
    if avg_risk > 0.5:
        st.markdown("### 🚨 **Final Verdict: High Cardiovascular Risk!**")
        st.markdown("💡 Please consult a doctor and take necessary precautions.")
    else:
        st.markdown("### ✅ **Final Verdict: Low Cardiovascular Risk!**")
        st.markdown("💪 Keep up the healthy lifestyle! 🚴‍♂️🥦")

