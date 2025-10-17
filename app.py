# app.py
# Streamlit app for Customer Churn Prediction (no EDA, no encoding needed)

import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load model and dataset
# ---------------------------
model = pickle.load(open("model (1).sav", "rb"))
df = pd.read_csv("tel_churn.csv")  # Adjust filename if different

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Provide customer details below and predict if they are likely to churn or continue.")

# ---------------------------
# Input Fields
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, value=12)

# ---------------------------
# Prediction Logic
# ---------------------------
if st.button("🔍 Predict"):
    input_data = pd.DataFrame([[
        SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents,
        PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, tenure
    ]], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'tenure'
    ])

    # ⚠️ Make sure your model expects categorical variables in this form.
    # If they were encoded numerically before training, you should encode them here too.

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1][0] * 100

    # Display results
    st.subheader("🧾 Prediction Result:")
    if prediction[0] == 1:
        st.error("This customer is **likely to churn** ⚠️")
        st.write(f"**Confidence:** {probability:.2f}%")
    else:
        st.success("This customer is **likely to continue** ✅")
        st.write(f"**Confidence:** {probability:.2f}%")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")

