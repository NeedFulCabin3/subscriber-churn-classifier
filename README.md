# Subscriber Churn Classifier

> Predicting customer retention outcomes through a structured machine learning pipeline using scikit-learn.

## Overview

Analyzing customer churn often leads to fragmented codebases where data preprocessing, transformation, and model training steps are scattered across ad-hoc scripts. This causes data leakage risks and inconsistent feature scaling during inference. `subscriber-churn-classifier` solves this operational friction by encapsulating the entire data preprocessing and logistic regression workflow inside a unified scikit-learn `Pipeline` and `ColumnTransformer` architecture.

## How It Works

The engine operates through a deterministic execution pipeline:
1. **Synthetic Data Generation:** Simulates realistic subscription metrics including tenure length, monthly billing charges, contract types, and payment methods with a controlled baseline churn distribution.
2. **Stratified Splitting:** Divides the dataset into training and testing subsets while maintaining class distribution proportions via `stratify=y`.
3. **Column-Specific Transformation:** Applies `StandardScaler` to continuous numeric columns (`tenure_months`, `monthly_charges`) and `OneHotEncoder` (with unknown category handling) to categorical attributes (`contract_type`, `payment_method`).
4. **Model Training & Evaluation:** Fits a `LogisticRegression` classifier within the pipeline wrapper, followed by direct extraction of evaluation metrics via `confusion_matrix` and `classification_report`.

## Key Features

* **Pipeline Encapsulation:** Combines preprocessing steps and classification estimators into a single object to prevent data leakage.
* **Mixed Data Handling:** Automatically routes numeric and categorical features through distinct transformation paths.
* **Interpretable Coefficients:** Leverages logistic regression to provide clear insights into feature weight contributions.
* **Comprehensive Metrics:** Outputs exact confusion matrices and precision-recall metrics for retention and churn classes.

## Tech Stack & Core Dependencies Breakdown

* **Python 3.x:** Core runtime environment.
* **Pandas:** Tabular data manipulation and synthetic data framing.
* **NumPy:** Vectorized array operations and random seed management.
* **Scikit-Learn:** Machine learning estimators, preprocessing transformers, and evaluation metrics.

## Environment & Web-Based Quick Start

You can run and modify this repository directly inside GitHub Codespaces or your preferred browser-based environment without local terminal setup.

### 1. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 3. Execute the Pipeline
```bash
python main.py
```

## Repository Structure
```bash
subscriber-churn-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
└── main.py
```

## Roadmap

**[ ]Type Stubs & Annotations:** Introduce strict Python type hints across all utility functions and transformer boundaries.

**[ ]Hyperparameter Tuning:** Integrate GridSearchCV to optimize regularization parameters for the logistic regression estimator.

**[ ]Experiment Tracking:** Add telemetry logging to record evaluation metrics across training runs.
