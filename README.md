# Customer Churn Predictor (End-to-End ML Project)

**Problem:** A telecom company wants to know in advance which customer is about to leave (churn), so it can step in with a timely offer and retain them.
**Type:** Binary Classification (Churn = 1, Stay = 0)
**Dataset:** IBM Telco Customer Churn (7043 customers, 20 features). Already available in `data/churn.csv`.

## Folder structure
```
churn_project/
├── data/churn.csv            # dataset
├── train.py                  # full ML pipeline (training)
├── app.py                    # Streamlit web app
├── requirements.txt
└── models/                   # created by train.py (model, graphs, metrics)
```

## How to run
```bash
pip install -r requirements.txt
python train.py              # ~1 min: compares + tunes + saves models
streamlit run app.py         # the app opens in your browser
```
**In Google Colab:** upload the files, `!pip install streamlit`, `!python train.py`. (Streamlit doesn't display directly in Colab, so run the app on your laptop.)

## Results (test set, unseen data)
| Metric | Value |
|---|---|
| ROC-AUC | ~0.85 |
| Recall (churn) | ~0.74 |
| Precision (churn) | ~0.56 |
| F1 | ~0.63 |

Model comparison (5-fold CV): Gradient Boosting and Logistic Regression are almost equal (AUC ~0.85), with Random Forest slightly behind. Gradient Boosting was tuned and became the final model.

---

## The 8 steps of the project (the flow of train.py)

| Step | What was done | Why |
|---|---|---|
| 1 | Converted `TotalCharges` to numeric, dropped `customerID`, made the target 0/1 | Blank spaces had turned the column into text; an ID has nothing to teach the model |
| 2 | Split train/test **first** (stratified) | To avoid data leakage. The test set is locked in a vault |
| 3 | Pipeline: impute → scale (numeric), impute → one-hot (categorical) | A model only understands numbers; a pipeline prevents leakage |
| 4 | Compared 4 models with 5-fold CV | One split can give a lucky score; the CV average is more trustworthy |
| 5 | Tuning with RandomizedSearchCV | To find the best hyperparameters |
| 6 | Threshold ~0.33 instead of 0.5 (on train out-of-fold predictions) | Missing a churner is expensive, so recall was increased |
| 7 | Final evaluation on the test set **once** | An honest score |
| 8 | Saved the pipeline with `joblib` | So the same preprocessing is applied automatically in the app |

## Important concepts (what you learn from this project)

**1. Class imbalance:** Only 26.5% of customers churn. If you say "Stay" for everyone, you get 73.5% accuracy! So accuracy is not reliable. We look at Recall, F1, and ROC-AUC. How we handle it: `class_weight="balanced"`, threshold tuning.

**2. Data leakage:** Any information from test/validation data getting into training. Wrong: `StandardScaler.fit()` on the whole data, then split. Right: split first, fit the scaler only on train. A `Pipeline` does this automatically.

**3. Precision vs Recall (you'll understand this through this project):**
- **Recall** = out of all the real churners, how many we caught (275 / 374 = 73.5%).
- **Precision** = out of everyone we labelled as churn, how many really churned (275 / 494 = 55.7%).
- In business, a retention offer is cheap and losing a customer is expensive, so recall is preferred.

**4. Confusion matrix:** `[[816 219], [99 275]]` → 816 correct "stay", 219 false alarms (FP), 99 **missed churners (FN)**, 275 correctly caught (TP).

**5. Threshold:** The model gives a probability. Lower threshold = more churners caught but more false alarms. This is a business decision.




## Practice (do it yourself, this is where coding skill comes from)
1. Add a new feature in `train.py`: `df["num_services"]` (how many add-on services the customer has taken).
2. Add `XGBoost` to the models (`pip install xgboost`) and compare.
3. Compare the confusion matrix at threshold 0.5 vs 0.33 and write down what changed.
4. Add one more tab in `app.py` that shows the top churn reasons.
