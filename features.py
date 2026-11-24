import numpy as np, pandas as pd

# Removed after testing (no loss in cross-validated performance): gender, Partner, DeviceProtection,
# OnlineBackup, StreamingTV, StreamingMovies. See the feature-selection note in the notebook.
SERVICE_COLS = ["OnlineSecurity", "TechSupport"]     # the add-ons we keep

NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
            "charge_gap", "num_support_services", "is_month_to_month", "autopay"]

CAT_COLS = ["Dependents", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "TechSupport", "Contract", "PaperlessBilling",
            "PaymentMethod", "tenure_group"]

# The raw columns a customer record must contain (the app and batch upload check for these)
REQUIRED_COLUMNS = ["SeniorCitizen", "Dependents", "tenure", "PhoneService", "MultipleLines",
                    "InternetService", "OnlineSecurity", "TechSupport", "Contract",
                    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]

def prepare(X):
    """Raw customer rows in, model-ready columns out. Runs identically at train and predict time."""
    X = X.copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    X = X.replace({"No internet service": "No", "No phone service": "No"})
    X["avg_monthly_spend"] = (X["TotalCharges"] / X["tenure"].replace(0, np.nan)).fillna(X["MonthlyCharges"])
    X["charge_gap"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
    X["num_support_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
    X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
    X["autopay"] = X["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    X["tenure_group"] = pd.cut(X["tenure"], [-1, 12, 24, 48, 1000],
                               labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    return X
