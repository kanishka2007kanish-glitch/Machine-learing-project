from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline = joblib.load(os.path.join(BASE_DIR, "models", "churn_pipeline.pkl"))

@app.route("/")
def home():
    return {"message": "Flask API running. POST /predict"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prob = pipeline.predict_proba(df)[0][1]
    pred = int(prob > 0.5)

    return jsonify({
        "churn_prediction": pred,
        "churn_probability": round(float(prob), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
