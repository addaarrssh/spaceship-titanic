import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
)


HIGH_RISK_EXAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
}

LOW_RISK_EXAMPLE = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
}


def load_model():
    with open("churn_logistic_model.pkl", "rb") as file:
        return pickle.load(file)


def set_example(example):
    for key, value in example.items():
        st.session_state[key] = value


try:
    model = load_model()
except Exception as exc:
    st.error(f"Error loading model: {exc}")
    st.stop()


st.title("Customer Churn Prediction App")
st.success("Model loaded successfully.")
st.write("Enter customer details below to predict whether the customer is likely to churn.")

st.markdown(
    """
    <style>
    .result-box {
        padding: 1rem 1.1rem;
        border-radius: 0.8rem;
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-churn {
        background-color: #4a1010;
        border: 1px solid #ff4b4b;
        color: #ffd8d8;
    }
    .result-stay {
        background-color: #10361d;
        border: 1px solid #2ecc71;
        color: #d7ffe4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Load Churn Example"):
        set_example(HIGH_RISK_EXAMPLE)
with col2:
    if st.button("Load No-Churn Example"):
        set_example(LOW_RISK_EXAMPLE)

threshold = st.slider("Churn Decision Threshold", 0.05, 0.90, 0.15, 0.01)

gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], key="SeniorCitizen")
Partner = st.selectbox("Partner", ["Yes", "No"], key="Partner")
Dependents = st.selectbox("Dependents", ["Yes", "No"], key="Dependents")
tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
PhoneService = st.selectbox("Phone Service", ["Yes", "No"], key="PhoneService")
MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"],
    key="MultipleLines",
)
InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
    key="InternetService",
)
OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"],
    key="OnlineSecurity",
)
OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"],
    key="OnlineBackup",
)
DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"],
    key="DeviceProtection",
)
TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"],
    key="TechSupport",
)
StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"],
    key="StreamingTV",
)
StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"],
    key="StreamingMovies",
)
Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"],
    key="Contract",
)
PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"],
    key="PaperlessBilling",
)
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    key="PaymentMethod",
)
MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0, key="MonthlyCharges")

TotalCharges_raw = tenure * MonthlyCharges
TotalCharges = np.log1p(TotalCharges_raw)

st.info(f"Estimated Total Charges used by model: ${TotalCharges_raw:.2f}")

input_df = pd.DataFrame(
    {
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
    }
)

st.subheader("Current Model Input")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Churn"):
    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.metric("Churn Probability", f"{probability:.2%}")
    st.metric("Current Threshold", f"{threshold:.2f}")

    if prediction == 1:
        st.markdown(
            "<div class='result-box result-churn'>This customer is likely to churn.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='result-box result-stay'>This customer is likely to stay.</div>",
            unsafe_allow_html=True,
        )
