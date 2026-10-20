"""
Loan Approval Prediction Pipeline
CodSoft ML Internship — Manas Mishra

End-to-end ML pipeline: EDA, preprocessing, model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.impute import SimpleImputer

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "Loan_Train.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load and return the raw loan dataset."""
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:\n{df['Loan_Status'].value_counts()}")
    return df


# ── 2. EDA ───────────────────────────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    """Print basic EDA stats."""
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n--- Numeric Summary ---")
    print(df.describe())


# ── 3. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    - Drop Loan_ID
    - Impute missing values (median for numeric, mode for categorical)
    - Label encode categorical features
    - Return X, y
    """
    df = df.drop(columns=["Loan_ID"])

    # Impute
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Loan_Status"]

    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    y = le.fit_transform(df["Loan_Status"])  # Y=1, N=0
    X = df.drop(columns=["Loan_Status"])

    print(f"\nFeatures after preprocessing: {list(X.columns)}")
    return X, y


# ── 4. Train & Evaluate ───────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test) -> dict:
    """Train Logistic Regression, Random Forest, and Gradient Boosting. Return results dict."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        X_tr = X_train_scaled if name == "Logistic Regression" else X_train
        X_te = X_test_scaled if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {"accuracy": acc, "f1": f1, "auc": auc, "model": model}
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    return results


# ── 5. Hyperparameter Tuning ──────────────────────────────────────────────────
def tune_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    """GridSearchCV on Gradient Boosting."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(gb, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_


# ── 6. Feature Importance Plot ────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, title: str = "Feature Importance") -> None:
    """Bar plot of feature importances."""
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Feature importance plot saved to feature_importance.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    explore_data(df)

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = train_models(X_train, X_test, y_train, y_test)

    print("\n\n=== Hyperparameter Tuning: Gradient Boosting ===")
    best_gb = tune_gradient_boosting(X_train, y_train)
    y_pred_best = best_gb.predict(X_test)
    print(f"Tuned GB Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

    plot_feature_importance(best_gb, list(X.columns), "Gradient Boosting — Feature Importance")

    print("\n✅ Pipeline complete.")
