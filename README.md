# 📉 Customer Churn Predictor

### End-to-End Machine Learning | Classification | Streamlit App

[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://customer-churn-predictor-cbcxhm59v4bmunbyfwv8p5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end machine learning project that predicts how likely a telecom customer is to leave (churn), so the business can offer a retention deal in time.

**🚀 Live app:** https://customer-churn-predictor-cbcxhm59v4bmunbyfwv8p5.streamlit.app/

<!-- Add your screenshots here. Create a folder called "screenshots" in the repo, upload the images, then keep the lines below. -->
<!--
![Single prediction](screenshots/single_prediction.png)
![Model performance](screenshots/model_performance.png)
-->

---

## 📌 Project Overview

| | |
|---|---|
| **Problem type** | Binary classification (Churn = 1, Stay = 0) |
| **Dataset** | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Size** | 7,043 customers, 19 input features (after dropping `customerID`) |
| **Final model** | Gradient Boosting Classifier (tuned) |
| **Class balance** | About 26.5% of customers churn |

## ✨ Features

- 🔮 **Single prediction:** enter one customer's details and get a churn probability.
- 📂 **Batch prediction:** upload a CSV and download churn probabilities for every customer.
- 🚦 **Risk levels:** Low, Medium and High, based on the predicted probability.
- 💡 **Retention suggestions:** simple rule-based ideas for customers flagged as at risk.
- 📊 **Performance tab:** metrics, model comparison, confusion matrix, ROC curve and feature importance.

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, Streamlit, Git and GitHub.

## 📂 Project Structure

```
Customer-Churn-Predictor/
├── app.py                 # Streamlit web app
├── train.py               # full training pipeline
├── requirements.txt
├── README.md
├── data/
│   └── churn.csv
└── models/
    ├── churn_pipeline.joblib
    ├── meta.json
    ├── model_comparison.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

## 🧠 Machine Learning Workflow

1. **Data cleaning:** converted `TotalCharges` to numeric (blank spaces had made it text), dropped `customerID`, and converted the target to 0/1.
2. **Train/test split:** 80/20, stratified, done **before** any preprocessing so test data never influences training.
3. **Preprocessing pipeline:** median imputation + `StandardScaler` for numeric columns; most-frequent imputation + `OneHotEncoder` for categorical columns, combined with `ColumnTransformer` and `Pipeline`.
4. **Model comparison:** Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, compared with 5-fold stratified cross-validation.
5. **Hyperparameter tuning:** `RandomizedSearchCV` on the best model, using ROC-AUC.
6. **Threshold selection:** chose the decision threshold (about 0.33 instead of the default 0.5) that maximizes F1 on out-of-fold predictions from the training data.
7. **Final evaluation:** scored once on the held-out test set (1,409 customers).
8. **Saving:** the full pipeline (preprocessing + model) is saved with Joblib, so the app applies the exact same preprocessing to new data.

## 📊 Model Performance

Results on the unseen test set:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.847 |
| Recall (churn) | 0.733 |
| Precision (churn) | 0.556 |
| F1-score (churn) | 0.632 |

**Confusion matrix (test set, threshold ≈ 0.33):**

|  | Predicted Stay | Predicted Churn |
|---|---:|---:|
| **Actually Stay** | 816 | 219 |
| **Actually Churn** | 100 | 274 |

In plain words: the model catches about 73 out of every 100 customers who really churn, but about 44 out of every 100 customers it flags would not have churned (false alarms).

**Effect of the threshold:** with the default 0.5 threshold, recall was about 0.53 and precision about 0.68. Lowering the threshold to about 0.33 raised recall to 0.73 and lowered precision to 0.56. This is a deliberate trade-off: a retention offer is usually cheaper than losing a customer, so catching more churners is worth some false alarms.

**Why not accuracy?** Only about 26.5% of customers churn, so a model that says "Stay" for everyone would get about 73.5% accuracy and catch zero churners. That is why this project focuses on recall, precision, F1 and ROC-AUC.

## ⚠️ Limitations

- The model is trained on one public dataset, so it may not match the behaviour of a real company's customers.
- The threshold maximizes the F1-score. It is not based on real retention costs or customer lifetime value. A real business should pick the threshold using its own costs.
- The retention suggestions are hand-written rules, not something the model learned. Treat them as ideas to investigate, not proven ways to stop churn.
- The model predicts risk, not certainty. A "flagged" customer is someone worth a closer look, not someone who will definitely leave.

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ayushpandey6714-svg/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) retrain the model
python train.py

# 5. Launch the app
streamlit run app.py
```

The trained model is already included in `models/`, so step 4 is optional. Running `train.py` regenerates the model, metrics and graphs.

**Batch CSV format:** your file must contain the same 19 feature columns as the training data. `customerID` and `Churn` are optional and ignored.

## 🔮 Future Improvements

- Add XGBoost and LightGBM to the model comparison.
- Add SHAP for explaining individual predictions.
- Add feature engineering (for example, number of add-on services per customer).
- Calibrate the predicted probabilities.
- Choose the threshold using real retention costs.

## 📄 Dataset Credit

IBM Telco Customer Churn dataset, publicly available on Kaggle.

## 👨‍💻 Author

**Ayush Pandey**
B.Tech, Electronics and Communication Engineering
Interested in Artificial Intelligence, Machine Learning and Software Development.

- **GitHub:** [@ayushpandey6714-svg](https://github.com/ayushpandey6714-svg)
- **Live demo:** [Try the app](https://customer-churn-predictor-cbcxhm59v4bmunbyfwv8p5.streamlit.app/)

⭐ If you find this project useful, consider giving it a star!
