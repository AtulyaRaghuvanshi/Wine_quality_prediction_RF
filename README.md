# Wine_quality_prediction_RF
# 🍷 Wine Quality Classification with Random Forests & Nested CV

This project implements a robust machine learning pipeline to classify red wine quality using **Random Forests**, **RFECV** for feature selection, and **nested cross-validation** for model evaluation and hyperparameter tuning. The entire process uses the **Wine Quality Dataset** from the UCI Machine Learning Repository.

---

## Project Objective

To build a binary classification model that predicts whether a red wine is of good quality (≥6) or not (<6), using Random Forests while optimizing for model robustness, feature selection, and generalization performance.

---

## Dataset Description

We use the `winequality-red.csv` dataset, which contains 1599 samples of red wine with 11 physicochemical input variables:

- `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`  
- `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`

Target variable:  
- `quality` (integer score between 3 and 8)

The dataset is converted to **binary classification**:
- Quality ≥ 6 → **Good wine (1)**
- Quality < 6 → **Bad wine (0)**

---

## Exploratory Data Analysis (EDA)

This includes:
- Missing value check
- Descriptive statistics
- Class distribution visualization
- Feature-target bar plots
- Heatmap of correlation matrix

---

## Feature Selection and Model Training

We use a **nested cross-validation strategy** with **Random Forests** for both:

1. Feature selection via `RFECV`
2. Final prediction via `RandomForestClassifier`

### Steps:

#### 1. RFECV (Recursive Feature Elimination with CV)
- Uses Random Forest to select the top `k` features from the 11 available.
- Evaluated using internal 5-fold CV on training folds.

#### 2. Grid Search
Performs hyperparameter tuning with:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}
```

##  Evaluation Strategy & Results

###  Outer Stratified K-Fold

- **5-fold stratified cross-validation** is used to ensure robust generalization testing.
- Prevents **data leakage** between feature selection and model evaluation by performing feature selection within each training fold.

---

###  Metrics Evaluated

For each fold and feature set size, the following metrics are computed:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1 Score (weighted)  
- ROC AUC  
- Specificity (TNR)

---

###  Output Files

Two Excel files are saved after evaluation:

- `train_results_RF_Wine.xlsx` — Metrics on training folds  
- `test_results_RF_Wine.xlsx` — Metrics on test/validation folds  

Each file contains the following columns:
Top Features, Selected Features, Fold, Accuracy, Precision, Recall, F1 Score, AUC, Specificity, Best Parameters


---

### Design Principles

-  No leakage via nested CV  
-  Hyperparameter optimization inside each outer fold  
-  Model interpretability through feature ranking  
-  Reproducibility with fixed random seeds

