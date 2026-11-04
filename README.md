# ML Assignment 2 — Classification Model Comparison

**Course:** M.Tech (AIML) — Machine Learning  
**Name:** Ashpreet Singh  
**ID:** 2026AJ0S708  
**Submission Date:** November 2026

---

## a. Problem Statement

The goal of this project is to implement and compare five classical machine learning classification models on a real-world medical diagnosis dataset. The models are trained to classify breast tumors as **malignant** or **benign** based on 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses. The project also includes an interactive **Streamlit web application** for model evaluation and comparison, deployed on Streamlit Community Cloud.

---

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (also available via `sklearn.datasets.load_breast_cancer()`)  
**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
**Original Creators:** Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin, 1995)

The Breast Cancer Wisconsin (Diagnostic) dataset is a widely used benchmark dataset in the machine learning community for binary classification tasks. It was created from digitized images of fine needle aspirate (FNA) biopsies of breast masses collected at the University of Wisconsin Hospitals, Madison. The dataset is used to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on characteristics of cell nuclei visible in the FNA images. This makes it a clinically relevant dataset where accurate classification can support early detection and treatment planning.

| Property | Value |
|---|---|
| **Number of Instances** | 569 |
| **Number of Features** | 30 |
| **Target Classes** | 2 — Malignant (0), Benign (1) |
| **Class Distribution** | 212 Malignant (37.3%), 357 Benign (62.7%) |
| **Missing Values** | None |
| **Feature Types** | All continuous (float64) |

For each cell nucleus image, 10 real-valued properties are measured: radius (mean distance from center to perimeter points), texture (standard deviation of gray-scale values), perimeter, area, smoothness (local variation in radius lengths), compactness (perimeter squared / area - 1.0), concavity (severity of concave portions of the contour), concave points (number of concave portions of the contour), symmetry, and fractal dimension (coastline approximation - 1). For each of these 10 properties, the **mean**, **standard error**, and **worst** (mean of the three largest values) are computed, giving 30 features in total. The features are computed using the Xcyt program, which performs precise boundary detection of cell nuclei using an interactive graphical interface.

The dataset has a moderate class imbalance (62.7% benign vs 37.3% malignant), which is typical of real-world medical datasets where benign cases tend to outnumber malignant ones. All features are continuous numeric values with no missing data, making it well-suited for standard classification algorithms without extensive preprocessing.

**Train-Test Split:** 80% training (455 samples), 20% testing (114 samples), stratified by target class, `random_state=42`. Stratification ensures both training and test sets maintain the same class proportions as the original dataset.

**Preprocessing:** StandardScaler fitted on training data only, then applied to both training and test sets. This prevents data leakage from the test set into the training process.

---

## c. GitHub Repository Link

**Repository:** [https://github.com/ashpreettsinghh/ml-classification-project](https://github.com/ashpreettsinghh/ml-classification-project)

**Repository Structure:**
```
ml-classification-project/
├── app.py                  # Streamlit web application
├── train_models.py         # Model training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test dataset (114 samples, 31 columns)
└── model/                  # Saved trained models and artifacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── scaler.joblib
    ├── feature_names.joblib
    ├── label_names.joblib
    └── metrics_summary.csv
```

**Live Streamlit App:** [https://ml-classification-project-2026aj0s708.streamlit.app](https://ml-classification-project-2026aj0s708.streamlit.app)

---

## d. Models Used

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieves the highest accuracy (98.25%) and F1 score (0.9825) on this dataset. Despite being a simple linear model, it performs exceptionally well because the Breast Cancer dataset has features that are largely linearly separable after standard scaling. The high AUC (0.9954) confirms excellent discrimination between malignant and benign classes. Logistic Regression also benefits from strong interpretability — the model coefficients directly indicate feature importance. |
| **Decision Tree** | Records the lowest performance among all five models with 91.23% accuracy and an MCC of 0.8174. Decision Trees are prone to overfitting on training data, which explains the relatively weaker generalization to the test set. The lower AUC (0.9157) compared to other models suggests the probability calibration is less reliable. While Decision Trees offer intuitive interpretability through their tree structure, they lack the regularization that other models benefit from. |
| **KNN** | Achieves a solid accuracy of 95.61% with an AUC of 0.9788. KNN is a non-parametric method that classifies samples based on the majority vote of their k nearest neighbors. Its performance benefits significantly from feature scaling (StandardScaler), as KNN is distance-based. With k=5, it strikes a reasonable balance between bias and variance. However, it falls short of Logistic Regression and SVM, possibly because the decision boundary in this feature space is better captured by linear or kernel-based methods. |
| **Naive Bayes** | Achieves 92.98% accuracy but a notably high AUC of 0.9868, which is the third-highest among all models. The gap between accuracy and AUC indicates that while the probability rankings are well-calibrated, the default decision threshold (0.5) may not be optimal. The conditional independence assumption of Naive Bayes is violated in this dataset (many features are correlated, e.g., radius, perimeter, and area), which limits its accuracy. However, its computational efficiency and reasonable probability estimates make it a useful baseline. |
| **Support Vector Machine** | Ties with Logistic Regression for the best accuracy (98.25%) and F1 score (0.9825). SVM with an RBF kernel can capture complex, non-linear decision boundaries, but on this dataset, the performance is nearly identical to Logistic Regression, suggesting the boundary is close to linear. The AUC (0.9950) is marginally lower than Logistic Regression (0.9954), but the difference is negligible. SVM's strength lies in its margin-maximization objective, which provides robust generalization. |
| **Overall Winner** | **Logistic Regression** is the winner for this dataset. It achieves the highest scores across all metrics (Accuracy: 98.25%, AUC: 0.9954, F1: 0.9825, MCC: 0.9623) while being the simplest and most interpretable model. SVM is a very close second with identical accuracy, but Logistic Regression edges ahead in AUC and offers better interpretability and faster inference. For a clinical diagnosis application, Logistic Regression provides the best trade-off between performance, interpretability, and computational efficiency. |

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ashpreettsinghh/ml-classification-project.git
cd ml-classification-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train models (optional — pre-trained models are included)
```bash
python train_models.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

### 5. Upload test data
Upload the provided `test_data.csv` file through the sidebar to see evaluation results.

---

## Streamlit App Features

- **CSV Upload:** Upload test data (CSV format with all features + target column)
- **Model Selection:** Dropdown to select any of the 5 classifiers for detailed analysis
- **Metrics Display:** All 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) displayed as metric cards
- **Comparison Table:** Side-by-side comparison of all models with best values highlighted
- **Bar Chart:** Visual comparison of metrics across all models
- **Confusion Matrix:** Heatmap visualization for the selected model
- **Classification Report:** Detailed per-class precision, recall, and F1 scores

---

## Technologies Used

- Python 3.12
- scikit-learn 1.5.2
- Streamlit 1.52
- pandas, numpy, matplotlib, seaborn
- joblib (model serialization)
