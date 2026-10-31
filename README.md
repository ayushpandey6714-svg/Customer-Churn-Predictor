# 📉 Customer Churn Predictor

An End-to-End Machine Learning Project that predicts whether a telecom customer is likely to leave a company (churn) using customer demographics, subscription details, and service information.

The project includes data preprocessing, model comparison, hyperparameter tuning, threshold optimization, model evaluation, and an interactive Streamlit web application.

## 🚀 Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing customers directly impacts revenue.

This project uses Machine Learning to identify customers who are likely to discontinue their telecom services, helping businesses take proactive retention measures.

**Problem Type:** Binary Classification  
**Target:** Churn (1 = Yes, 0 = No)  
**Dataset:** IBM Telco Customer Churn Dataset (7,043 customers)

## ✨ Features

- **Single Customer Prediction:** Predict churn probability using customer details.
- **Batch Prediction:** Upload a CSV file to predict churn for multiple customers.
- **Risk Classification:** Categorize customers into Low, Medium, and High risk.
- **Retention Suggestions:** Provide rule-based suggestions for customers at risk.
- **Model Performance Dashboard:** View evaluation metrics, confusion matrix, ROC curve, and feature importance.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas & NumPy | Data Cleaning and Processing |
| Scikit-learn | Machine Learning |
| Matplotlib | Data Visualization |
| Joblib | Model Serialization |
| Streamlit | Interactive Web Application |
| Git & GitHub | Version Control |

## 📂 Project Structure

```text
Customer-Churn-Predictor/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── churn.csv
│
└── models/
    ├── churn_pipeline.joblib
    ├── meta.json
    ├── model_comparison.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ayushpandey6714-svg/Customer-Churn-Predictor.git

cd Customer-Churn-Predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Model

```bash
python train.py
```

This script performs data preprocessing, model comparison, hyperparameter tuning, threshold selection, and model evaluation.

It also saves the trained model and evaluation artifacts inside the `models/` directory.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning pipeline consisting of eight major steps:

1. **Data Cleaning:** Converted `TotalCharges` into numeric format, handled missing values, removed unnecessary customer IDs, and encoded the target variable.

2. **Train-Test Split:** Used stratified splitting to maintain the same churn distribution in training and testing datasets.

3. **Data Preprocessing:** Used `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` through a `ColumnTransformer` and `Pipeline`.

4. **Model
