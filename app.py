import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("bank_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Bank Marketing Prediction", page_icon="🏦", layout="wide")

st.title("🏦 Bank Marketing Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", [
        "management", "blue-collar", "technician", "admin.", "services",
        "retired", "self-employed", "student", "unemployed",
        "entrepreneur", "housemaid", "unknown"
    ])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["secondary", "tertiary", "primary", "unknown"])

with col2:
    balance = st.number_input("Account Balance", value=1000)
    default = st.selectbox("Credit Default?", ["no", "yes"])
    housing = st.selectbox("Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Personal Loan?", ["no", "yes"])

with col3:
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    day = st.number_input("Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    duration = st.number_input("Call Duration (seconds)", min_value=0, value=300)

st.subheader("Campaign Information")

col4, col5, col6 = st.columns(3)

with col4:
    campaign = st.number_input("Number of Contacts in Current Campaign", min_value=1, value=2)

with col5:
    pdays = st.number_input("Days Since Last Contact (-1 if never contacted)", value=-1)

with col6:
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)

poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "success", "other"])

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day": [day],
    "month": [month],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]
})

# Encoding
input_data["default"] = input_data["default"].map({"no": 0, "yes": 1})
input_data["housing"] = input_data["housing"].map({"no": 0, "yes": 1})
input_data["loan"] = input_data["loan"].map({"no": 0, "yes": 1})

# Feature Engineering
input_data["contacted_before"] = input_data["pdays"].apply(lambda x: 0 if x == -1 else 1)
input_data["pdays"] = input_data["pdays"].replace(-1, 0)

input_data["high_campaign_contact"] = input_data["campaign"].apply(lambda x: 1 if x > 3 else 0)

# One-hot encoding
input_data = pd.get_dummies(input_data)

# Match training columns
input_data = input_data.reindex(columns=model_columns, fill_value=0)

st.divider()

if st.button("Predict Subscription"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("✅ The customer is likely to subscribe to the term deposit.")
    else:
        st.error("❌ The customer is not likely to subscribe to the term deposit.")

    st.metric("Subscription Probability", f"{probability * 100:.2f}%")