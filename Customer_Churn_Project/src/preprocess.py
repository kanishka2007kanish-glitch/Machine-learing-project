import pandas as pd

def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

def preprocess_data(df):
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

    # Encode target
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # Drop columns not needed
    df = df.drop(columns=["customerID"])

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df
