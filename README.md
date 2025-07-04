# 🧠 ANN-Based Customer Churn Prediction

This project is a Streamlit web application that predicts whether a customer is likely to churn, based on various input features. It uses an Artificial Neural Network (ANN) model trained on banking customer data.

🌐 **Live App**: [Click here to use it](https://ann-classification-churn-qzfwkumlbfn67saa5rhv6b.streamlit.app/)

---

## 🚀 Features

- Streamlit-based interactive web UI
- ANN model built using TensorFlow/Keras
- Real-time predictions based on user input
- Input preprocessing using:
  - Label Encoding
  - One-Hot Encoding
  - Standard Scaling

---

## 📦 Tech Stack

| Component         | Tool                  |
|------------------|-----------------------|
| Web Framework     | Streamlit             |
| ML Library        | TensorFlow (Keras)    |
| Preprocessing     | scikit-learn          |
| Data Handling     | Pandas, NumPy         |

---

## 🧠 Model Overview

The ANN model was trained on customer churn data using:
- Multiple dense layers
- ReLU activations
- Binary cross-entropy loss
- Adam optimizer

Model artifacts used:
- `model.h5` – Trained ANN model
- `label_enc.pkl` – Label encoder for gender
- `ohe.pkl` – OneHotEncoder for geography
- `scaler.pkl` – StandardScaler for numerical features

---


