import streamlit as st
import joblib
import pandas as pd

# Load trained objects
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction")
st.write("Upload customer data CSV to predict churn")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    customer_id = df["customerID"]
    df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    X = encoder.transform(df)
    X = scaler.transform(X)

    churn_pred = model.predict(X)
    churn_prob = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "Customer ID": customer_id,
        "Churn Prediction": churn_pred,
        "Churn Probability": churn_prob
    })

    st.subheader("Prediction Results")
    st.success("Prediction completed successfully ✅")
    st.dataframe(result)

    # Download option
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )