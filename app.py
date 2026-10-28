"""
Streamlit App — ML Classification Model Comparison
Course: M.Tech (AIML) — Machine Learning
Author: Ashpreet Singh (2026AJ0S708)

This app loads pre-trained classification models and evaluates them
on user-uploaded test data. No retraining is performed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
SCALER_FILE = "scaler.joblib"
FEATURE_NAMES_FILE = "feature_names.joblib"
LABEL_NAMES_FILE = "label_names.joblib"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_name: str):
    """Load a saved model from the model directory."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the saved StandardScaler."""
    return joblib.load(os.path.join(MODEL_DIR, SCALER_FILE))


@st.cache_resource
def load_feature_names():
    """Load the saved feature names."""
    return joblib.load(os.path.join(MODEL_DIR, FEATURE_NAMES_FILE))


@st.cache_resource
def load_label_names():
    """Load the saved label/class names."""
    return joblib.load(os.path.join(MODEL_DIR, LABEL_NAMES_FILE))


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all 6 required evaluation metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    # AUC — requires probability scores
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["AUC"] = roc_auc_score(y_true, prob)
            else:
                metrics["AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            metrics["AUC"] = float("nan")
    else:
        metrics["AUC"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_pred, label_names, model_name):
    """Generate a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("ML Classification Model Comparison")
    st.markdown(
        """
        **Course:** M.Tech (AIML) — Machine Learning  
        **Author:** Ashpreet Singh (2026AJ0S708)  
        **Dataset:** Breast Cancer Wisconsin (Diagnostic)
        
        Upload test data (CSV) and compare the performance of five classifiers:
        Logistic Regression, Decision Tree, KNN, Naive Bayes, and SVM.
        """
    )

    st.divider()

    # --- Sidebar: file upload and model selection ---
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader(
            "Upload Test Data (CSV)",
            type=["csv"],
            help="CSV must contain all input features and a 'target' column.",
        )
        model_choice = st.selectbox(
            "Select Model for Detailed View",
            list(MODEL_FILES.keys()),
        )
        compare_all = st.checkbox("Compare all models", value=True)

    # --- Load artifacts ---
    try:
        scaler = load_scaler()
        feature_names = load_feature_names()
        label_names = load_label_names()
    except FileNotFoundError as e:
        st.error(f"Missing model artifacts: {e}. Please ensure models are trained first.")
        st.stop()

    # --- Process uploaded file ---
    if uploaded_file is None:
        st.info("Upload a test CSV file from the sidebar to get started.")
        st.markdown("---")
        st.markdown("### Expected CSV Format")
        st.markdown(
            f"The CSV must include **{len(feature_names)} feature columns** "
            f"and a **`target`** column."
        )
        st.markdown("**Feature columns:** " + ", ".join(f"`{f}`" for f in feature_names))
        st.stop()

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Validate columns
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        st.error(f"Missing feature columns: {missing_features}")
        st.stop()
    if "target" not in df.columns:
        st.error("CSV must contain a `target` column.")
        st.stop()

    st.success(f"Loaded test data: {df.shape[0]} samples, {df.shape[1]} columns")

    X_test_raw = df[feature_names].values
    y_test = df["target"].values

    # Scale features using the saved scaler
    X_test = scaler.transform(X_test_raw)

    # ------------------------------------------------------------------
    # Section: Compare all models
    # ------------------------------------------------------------------
    if compare_all:
        st.header("All Models — Comparison")

        all_metrics = {}
        for name in MODEL_FILES:
            model = load_model(name)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            all_metrics[name] = compute_metrics(y_test, y_pred, y_prob)

        # Build comparison table
        comparison_df = pd.DataFrame(all_metrics).T
        # Reorder columns
        col_order = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
        comparison_df = comparison_df[col_order]

        st.dataframe(
            comparison_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
            use_container_width=True,
        )

        # Bar chart comparison
        st.subheader("Metric Comparison (Bar Chart)")
        fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
        comparison_df.plot(kind="bar", ax=ax_bar, width=0.75)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("Model Performance Comparison")
        ax_bar.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig_bar)

    # ------------------------------------------------------------------
    # Section: Selected model detail
    # ------------------------------------------------------------------
    st.header(f"Detailed View — {model_choice}")

    model = load_model(model_choice)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Metrics cards
    cols = st.columns(6)
    metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    for col, name in zip(cols, metric_names):
        value = metrics.get(name, float("nan"))
        col.metric(label=name, value=f"{value:.4f}")

    # Two-column layout: confusion matrix + classification report
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test, y_pred, label_names, model_choice)
        st.pyplot(fig_cm)

    with col_right:
        st.subheader("Classification Report")
        report = classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        "Built with Streamlit | Breast Cancer Wisconsin Dataset | "
        "Ashpreet Singh — 2026AJ0S708"
    )


if __name__ == "__main__":
    main()
