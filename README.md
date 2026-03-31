---
title: Credit Card Fraud Detection
emoji: "🛡️"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Fraud Detection System

This project builds a credit card fraud detection system using machine learning and deploys the final model with Streamlit.

The dataset is highly imbalanced, so the project focuses more on recall and ROC-AUC than plain accuracy.

## Project Objective

The goal is to classify a transaction as:

- `0` = Legitimate
- `1` = Fraudulent

This is a binary classification problem on the `creditcard.csv` dataset.

## Dataset Summary

- Rows: `284,807`
- Columns: `31`
- Target column: `Class`
- Fraud cases: `492`
- Legitimate cases: `284,315`
- Fraud ratio: `0.1727%`

The dataset mostly contains PCA-transformed numeric features (`V1` to `V28`) along with `Time`, `Amount`, and `Class`.

## Workflow Used

The notebook follows a practical fraud detection pipeline:

1. Data understanding and basic EDA
2. Class imbalance detection
3. SMOTE on training data only
4. Feature scaling for required models
5. Feature selection using model importance
6. Training multiple models
7. Hyperparameter tuning
8. Cross validation
9. ANN model implementation
10. Final model comparison and deployment

## Models Considered

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)

The final deployed model in this project is a `RandomForestClassifier`.

## Evaluation Focus

For fraud detection, these metrics are more important than accuracy:

- Precision
- Recall
- F1 Score
- ROC-AUC

Recall is especially important because missing a fraud case is more costly than flagging a normal case.

## Project Files

- `creditcard.csv` - dataset used locally for training and testing
- `Fraud_Detection_exam.ipynb` - main notebook with preprocessing, modeling, ANN, and comparison
- `app.py` - Streamlit deployment app
- `best_model.pkl` - saved final model
- `scaler.pkl` - saved scaler for input preprocessing
- `requirements.txt` - project dependencies

Note: the raw dataset is not pushed to GitHub because it exceeds standard GitHub file size limits. Download `creditcard.csv` separately and place it in the project root before running the notebook locally.

## How To Run

1. Install dependencies for the app

```bash
pip install -r requirements.txt
```

2. Install extra notebook libraries if needed

```bash
pip install matplotlib seaborn imbalanced-learn tensorflow kagglehub
```

3. Run the notebook

Open `Fraud_Detection_exam.ipynb` in Jupyter Notebook or JupyterLab and run the cells step by step.

4. Run the Streamlit app

```bash
streamlit run app.py
```

## Streamlit App Input

The app accepts:

- `Time`
- `Amount`
- `V1` to `V28`

After clicking the predict button, the app shows whether the transaction is:

- Legitimate
- Fraudulent

It also displays prediction probabilities.

## Why Random Forest Was Selected

- It handles nonlinear fraud patterns well
- It works well on numeric-heavy data
- It is strong on noisy data
- It gave reliable classification performance for this dataset
- It was better suited for deployment in this project

If ANN performance is lower than classical models, selecting Random Forest is still justified because final selection should depend on recall, ROC-AUC, and generalization.

## Short Viva Explanation

`SMOTE balances minority fraud samples so the model can learn fraud patterns better.`

`Recall is important because missing fraudulent transactions is more dangerous than a few false alarms.`

`Random Forest was selected because it handled the imbalanced numeric dataset well and gave strong fraud detection performance.`

## Deployment Note

The Streamlit app uses the saved model from `best_model.pkl` and scales the `Time` and `Amount` inputs using `scaler.pkl` before prediction.
