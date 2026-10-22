"""
Model Evaluation Script for Loan Approval Prediction.
Executes stratified evaluation across models and saves structured metrics JSON.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def preprocess_and_evaluate():
    df = pd.read_csv("Loan_Train.csv")
    df = df.drop(columns=["Loan_ID"])

    # Imputation
    num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    num_imp = SimpleImputer(strategy="median")
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # Feature Engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]

    # Encoding
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "f1_score": round(float(f1_score(y_test, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "cv_mean_accuracy": round(float(np.mean(cv_scores)), 4)
        }
        results[name] = metrics
        print(f"[{name}] Acc: {metrics['accuracy']} | F1: {metrics['f1_score']} | CV: {metrics['cv_mean_accuracy']}")

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Successfully exported metrics.json")
    return results

if __name__ == "__main__":
    preprocess_and_evaluate()
