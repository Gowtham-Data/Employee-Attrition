import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model :
with open("rf_attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    saved_encoders = pickle.load(f)

st.title("🧑‍💼 Employee Attrition Prediction (Minimal Inputs)")

selected_cols = [
    'Age', 'MonthlyIncome', 'YearsAtCompany',
    'OverTime', 'JobRole', 'MaritalStatus'
]

job_roles = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]

marital_status_options = ["Single", "Married", "Divorced"]
overtime_options = ["Yes", "No"]

# INPUT UI:
input_data = {}
input_data['Age'] = st.number_input("Age", 18, 65)
input_data['MonthlyIncome'] = st.number_input("Monthly Income", 1000, 50000)
input_data['YearsAtCompany'] = st.number_input("Years At Company", 0, 40)
input_data['OverTime'] = st.radio("OverTime?", overtime_options)
input_data['JobRole'] = st.selectbox("Job Role", job_roles)
input_data['MaritalStatus'] = st.selectbox("Marital Status", marital_status_options)

input_df = pd.DataFrame([input_data])

# Correct categorical encoding
for col in ['OverTime', 'JobRole', 'MaritalStatus']:
    le = saved_encoders[col]
    if input_df[col][0] in le.classes_:
        input_df[col] = le.transform(input_df[col])
    else:
        input_df[col] = 0

# Create correct feature structure
final_df = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
for col in selected_cols:
    final_df[col] = input_df[col]

# Scale numeric variables using SAME scaler
num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany']
final_df[num_cols] = scaler.transform(final_df[num_cols])

# Predict button :
if st.button("🔍 Predict Attrition"):
    proba = model.predict_proba(final_df)[0][1]
    prediction = int(proba >= 0.50)

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Employee likely to **Leave** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Employee likely to **Stay** (Probability: {proba:.2f})")
