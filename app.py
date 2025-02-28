import streamlit as st
import joblib
import numpy as np
import gdown
import os

# Download model from Google Drive
model_url = "https://drive.google.com/uc?id=1MPhXD6NRWBlHrR6tyuLwFpsphFEAFwVH"  # Replace with your File ID
model_path = "bw1_model.pkl"
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load the model
model_package = joblib.load(model_path)
teacher_model = model_package['teacher_model']
student_model = model_package['student_model']
scaler_X = model_package['scaler_X']
scaler_y = model_package['scaler_y']

expected_features = len(scaler_X.mean_)

st.title("West Floor 1 Moisture Prediction")
st.write(f"Enter {expected_features} feature values for BW1 (comma-separated) to predict moisture content.")

example_input = "238.39, 218.90, 14.64"  # Adjust based on your X.columns count
feature_input = st.text_input(f"Features (e.g., {example_input} for BW1-MCheek, BW1-MJoistUp, etc.)", example_input)

if st.button("Predict"):
    try:
        feature_list = [x.strip() for x in feature_input.split(',')]
        if len(feature_list) != expected_features:
            st.error(f"Expected {expected_features} features, but got {len(feature_list)}.")
        else:
            features = np.array([float(x) for x in feature_list]).reshape(1, -1)
            features_scaled = scaler_X.transform(features)
            teacher_pred = teacher_model.predict(features_scaled)
            student_pred = student_model.predict(features_scaled)
            final_pred_scaled = (teacher_pred + student_pred) / 2
            final_pred = scaler_y.inverse_transform(final_pred_scaled)
            st.success("Predictions (Moisture Content):")
            st.write(final_pred.tolist())
    except ValueError as e:
        st.error(f"Invalid input: {e}. Please enter numeric values only.")
    except Exception as e:
        st.error(f"Error: {e}")

st.write("Note: Feature order must match training data.")
