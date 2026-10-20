# Loan Approval Prediction

A machine learning pipeline that predicts loan approval decisions using supervised classification — built as part of the CodSoft ML Internship.

## Problem Statement

Financial institutions need to automate and standardize loan approval decisions. Given applicant demographics and financial profile, predict whether a loan will be approved (`Y`) or rejected (`N`).

## Dataset

- **Source:** [Loan Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Size:** 614 training samples, 13 features
- **Target:** `Loan_Status` (Y / N)

### Features

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Marital status |
| `Dependents` | Categorical | Number of dependents |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Employment type |
| `ApplicantIncome` | Numerical | Monthly income of applicant |
| `CoapplicantIncome` | Numerical | Monthly income of co-applicant |
| `LoanAmount` | Numerical | Loan amount requested (thousands) |
| `Loan_Amount_Term` | Numerical | Term in months |
| `Credit_History` | Binary | Credit history meets guidelines (1/0) |
| `Property_Area` | Categorical | Urban / Rural / Semiurban |

## Approach

1. **Exploratory Data Analysis** — distribution plots, correlation matrix, class balance
2. **Data Preprocessing** — median/mode imputation, label encoding, feature scaling
3. **Model Training** — Logistic Regression, Random Forest, Gradient Boosting comparison
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
5. **Hyperparameter Tuning** — GridSearchCV on the best-performing model

## Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.83 | ~0.79 |
| Random Forest | ~82% | ~0.85 | ~0.82 |
| Gradient Boosting | **~83%** | **~0.86** | **~0.84** |

> Gradient Boosting with tuned hyperparameters achieved the best overall performance.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python loan_approval.py
```

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)

## License

MIT — see [LICENSE](LICENSE)
