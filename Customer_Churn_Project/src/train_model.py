import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load data
df = pd.read_csv(DATA_PATH)

# Target
y = df["Churn"].map({"Yes": 1, "No": 0})

# Features
X = df.drop(columns=["Churn", "customerID"])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save model, scaler, encoder
joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))

print("✅ Training complete. Model, scaler, encoder saved.")
