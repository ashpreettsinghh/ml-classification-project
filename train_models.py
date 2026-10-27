"""
train_models.py — Train 5 classification models on the Breast Cancer dataset.

Course: M.Tech (AIML) — Machine Learning
Author: Ashpreet Singh (2026AJ0S708)

Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = "model"

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(
        probability=True, random_state=RANDOM_STATE
    ),
}

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def main():
    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Loading Breast Cancer Wisconsin dataset")
    print("=" * 70)

    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    label_names = list(data.target_names)

    print(f"  Features      : {len(feature_names)}")
    print(f"  Instances     : {X.shape[0]}")
    print(f"  Classes       : {label_names}")
    print(f"  Feature names : {feature_names[:5]} ... (showing first 5)")
    print()

    # ------------------------------------------------------------------
    # Step 2: Train-test split
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 2: Splitting data (80% train, 20% test)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Test samples  : {X_test.shape[0]}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Save test data as CSV (raw, unscaled)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 3: Saving test_data.csv")
    print("=" * 70)

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)
    print(f"  Saved: test_data.csv ({test_df.shape[0]} rows, {test_df.shape[1]} cols)")
    print()

    # ------------------------------------------------------------------
    # Step 4: Feature scaling
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 4: Fitting StandardScaler on training data")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaler fitted on training data, applied to both sets.")
    print()

    # ------------------------------------------------------------------
    # Step 5: Train models and evaluate
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 5: Training and evaluating models")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    results = []
    for name, model in MODELS.items():
        print(f"\n  Training: {name}")
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = (
            model.predict_proba(X_test_scaled)
            if hasattr(model, "predict_proba")
            else None
        )

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        if y_prob is not None:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = float("nan")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "MCC": round(mcc, 4),
        })

        # Save model
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAMES[name])
        joblib.dump(model, model_path)
        print(f"    Saved: {model_path}")
        print(f"    Accuracy={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Save scaler, feature names, label names
    # ------------------------------------------------------------------
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(label_names, os.path.join(MODEL_DIR, "label_names.joblib"))
    print(f"\n  Saved: scaler, feature_names, label_names")

    # ------------------------------------------------------------------
    # Step 7: Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: Model Performance Comparison")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Model")
    print(results_df.to_string())

    # Save metrics summary
    results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"))
    print(f"\n  Saved: {MODEL_DIR}/metrics_summary.csv")

    # ------------------------------------------------------------------
    # Step 8: Identify winner
    # ------------------------------------------------------------------
    winner = results_df["F1 Score"].idxmax()
    print(f"\n  Overall Winner (by F1 Score): {winner}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
