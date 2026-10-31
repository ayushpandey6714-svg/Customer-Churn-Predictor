# 📉 Customer Churn Predictor

### End-to-End Machine Learning | Classification | Streamlit Deployment

[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://customer-churn-predictor-cbcxhm59v4bmunbyfwv8p5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end Machine Learning application that predicts the probability of telecom customers leaving a company (customer churn).

This project demonstrates a complete ML workflow, from data preprocessing and model training to evaluation and deployment using Streamlit.

**🚀 Live Application:** https://customer-churn-predictor-cbcxhm59v4bmunbyfwv8p5.streamlit.app/

---

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing customers directly affects revenue and long-term business growth.

This project uses Machine Learning to identify customers who are likely to discontinue their telecom services.

The model analyzes customer demographics, subscription information, service usage, and billing details to estimate churn probability.

**Problem Type:** Binary Classification  
**Dataset:** IBM Telco Customer Churn  
**Dataset Size:** 7,043 customers  
**Target Variable:** Churn (Yes = 1, No = 0)  
**Final Model:** Gradient Boosting Classifier

## ✨ Key Features

- 🔮 **Single Customer Prediction:** Predict churn probability using customer details.
- 📂 **Batch Prediction:** Upload a CSV file to predict churn for multiple customers.
- 🚦 **Risk Classification:** Categorize customers into Low, Medium, and High risk.
- 💡 **Retention Suggestions:** Generate rule-based suggestions for customers at risk.
- 📊 **Model Performance Dashboard:** View evaluation metrics, model comparison, confusion matrix, ROC curve, and feature importance.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| Matplotlib | Data Visualization |
| Joblib | Model Serialization |
| Streamlit | Web Application |
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

## 🧠 Machine Learning Workflow

The project follows an eight-step machine learning pipeline:

**1. Data Cleaning**
- Converted `TotalCharges` into numeric format.
- Removed unnecessary `customerID`.
- Converted the target variable into binary values.
- Identified missing values.

**2. Train-Test Split**
- Used an 80:20 train-test split.
- Applied stratification to maintain the churn distribution.
- Prevented data leakage by splitting before preprocessing.

**3. Data Preprocessing**
- `SimpleImputer` for missing values.
- `StandardScaler` for numerical features.
- `OneHotEncoder` for categorical features.
- `ColumnTransformer` and `Pipeline` to combine preprocessing steps.

**4. Model Comparison**

Compared four classification algorithms using 5-Fold Stratified Cross-Validation:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

**5. Hyperparameter Tuning**
- Used `RandomizedSearchCV` to find better model configurations.

**6. Threshold Optimization**
- Selected a classification threshold using training out-of-fold predictions.
- Optimized the F1-score instead of blindly using the default 0.5 threshold.

**7. Model Evaluation**
- Evaluated the final model on an unseen test dataset.
- Used Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

**8. Model Serialization**
- Saved the complete preprocessing and trained model pipeline using Joblib.
- Loaded the saved artifacts in the Streamlit application.

## 📊 Model Performance

The final Gradient Boosting model was evaluated on an unseen test dataset.

| Metric | Approximate Score |
|---|---:|
| ROC-AUC | 0.85 |
| Recall | 0.74 |
| Precision | 0.56 |
| F1-Score | 0.63 |

**Why Recall Matters**

The dataset contains approximately 26.5% churn cases, making it an imbalanced classification problem.

Accuracy alone can be misleading because a model predicting every customer as "Stay" can achieve approximately 73.5% accuracy without identifying any actual churners.

Therefore, this project focuses on Recall, Precision, F1-Score, and ROC-AUC.

Recall is particularly important because identifying potential churners allows businesses to take preventive retention actions.

## 📚 Key Concepts Implemented

- Binary Classification
- Class Imbalance
- Data Leakage Prevention
- Stratified Train-Test Split
- Cross-Validation
- Feature Scaling
- One-Hot Encoding
- Scikit-learn Pipelines
- Hyperparameter Tuning
- Precision-Recall Trade-off
- Classification Threshold Optimization
- Model Evaluation
- Model Serialization
- Streamlit Deployment

## ⚙️ Run Locally

Want to run the project on your machine?

**1. Clone the repository**

```bash
git clone https://github.com/ayushpandey6714-svg/Customer-Churn-Predictor.git

cd Customer-Churn-Predictor
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Train the model**

```bash
python train.py
```

This generates the trained model, metadata, evaluation graphs, and model comparison results inside the `models/` directory.

**6. Launch the application**

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔮 Future Improvements

- Integrate XGBoost and LightGBM for additional model comparisons.
- Implement SHAP for explainable AI and individual prediction explanations.
- Add feature engineering to improve predictive performance.
- Explore probability calibration.
- Improve customer segmentation and retention insights.

## 👨‍💻 Author

**Ayush Pandey**

B.Tech – Electronics and Communication Engineering

Interested in Artificial Intelligence, Machine Learning, and Software Development.

- **GitHub:** [@ayushpandey6714-svg](https://github.com/ayushpandey6714-svg)
- **Project Repository:** [Customer Churn Predictor](https://github.com/ayushpandey6714-svg/Customer-Churn-Predictor)
- **Live Demo:** [Try the Application](https://customer-churn-predictor-cbcxhm59v4bmunbyfwv8p5.streamlit.app/)

---

⭐ If you find this project useful, consider giving it a star!
