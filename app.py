import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

st.title("Customer Churn Prediction")

try:
    # Define paths
    model_path = os.path.join("models", "model.h5")
    label_enc_path = os.path.join("models", "label_enc.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    ohe_path = os.path.join("models", "ohe.pkl")

    # Load model and encoders
    model = tf.keras.models.load_model(model_path)

    with open(label_enc_path, 'rb') as file:
        label_enc = pickle.load(file)

    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    with open(ohe_path, 'rb') as file:
        ohe = pickle.load(file)

    # User Inputs
    geography = st.selectbox('Geography', ohe.categories_[0])
    gender = st.selectbox('Gender', label_enc.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

    # Prepare input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_enc.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = ohe.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'📊 Churn Probability: {prediction_proba:.2f}')
    if prediction_proba > 0.5:
        st.error('🚨 The customer is likely to churn.')
    else:
        st.success('✅ The customer is not likely to churn.')

except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
