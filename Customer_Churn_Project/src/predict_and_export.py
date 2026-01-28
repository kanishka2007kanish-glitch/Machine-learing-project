import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "churn_predictions.csv")

# Load data
df = pd.read_csv(DATA_PATH)

customer_id = df["customerID"]

# Drop target
df = df.drop(columns=["Churn", "customerID"])

# Load encoder, scaler, model
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# Encode categorical features
X_encoded = encoder.transform(df)

# Scale
X_scaled = scaler.transform(X_encoded)

# Predict
churn_pred = model.predict(X_scaled)
churn_prob = model.predict_proba(X_scaled)[:, 1]

# Save output
output_df = pd.DataFrame({
    "customer_id": customer_id,
    "churn_prediction": churn_pred,
    "churn_probability": churn_prob
})

output_df.to_csv(OUTPUT_PATH, index=False)

print("✅ churn_predictions.csv created successfully")
