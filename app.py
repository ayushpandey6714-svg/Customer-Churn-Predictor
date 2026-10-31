"""
CUSTOMER CHURN PREDICTOR - STREAMLIT APP
=========================================
Run:  streamlit run app.py
(You must run `python train.py` first so that the model is created in the models/ folder.)

What is Streamlit? A tool for building web apps in pure Python. No HTML/JS needed.
Streamlit's rule: whenever the user changes any widget, the whole script reruns from top to bottom.
"""

import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------
# PAGE SETTINGS (must be the very first streamlit command)
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------
# MODEL LOAD
# @st.cache_resource: the model is loaded only ONCE, not on every click (keeps the app fast)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/churn_pipeline.joblib")     # preprocessing + model, both together
    with open("models/meta.json") as f:
        meta = json.load(f)                                # threshold, metrics, dropdown options
    return pipe, meta


if not Path("models/churn_pipeline.joblib").exists():
    st.error("Model file not found. Please run `python train.py` in the terminal first.")
    st.stop()                                              # do not run the code below

pipe, meta = load_artifacts()
THRESHOLD = meta["threshold"]
cat_opts = meta["categorical_options"]                     # {"Contract": ["Month-to-month", ...], ...}


def risk_label(p):
    """Convert a probability into a business-friendly label."""
    if p >= 0.6:
        return "HIGH", "🔴"
    if p >= THRESHOLD:
        return "MEDIUM", "🟠"
    return "LOW", "🟢"


def suggestion(row):
    """Simple rule-based retention tips (business logic, not the model)."""
    tips = []
    if row["Contract"] == "Month-to-month":
        tips.append("Month-to-month customers churn the most: offer a discount for a 1- or 2-year contract.")
    if row["InternetService"] == "Fiber optic":
        tips.append("Fiber optic users churn more: collect price/quality feedback and offer a loyalty discount.")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        tips.append("Offer a free/discounted Tech Support trial; customers with support churn less.")
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        tips.append("Offer a free trial of the Online Security add-on.")
    if row["PaymentMethod"] == "Electronic check":
        tips.append("Offer a small cashback for switching to auto-pay (card/bank transfer).")
    if row["tenure"] < 12:
        tips.append("New customer (<12 months): offer an onboarding call and a welcome offer.")
    return tips


# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption("Predict the probability that a telecom customer will leave (churn).")

tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)", "📊 Model Performance"])

# ===============================================================
# TAB 1: SINGLE CUSTOMER
# ===============================================================
with tab1:
    st.subheader("Enter the customer's details")

    # st.columns: splits the screen into 3 parts so the form doesn't look too long
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", cat_opts["gender"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", cat_opts["Partner"])
        dependents = st.selectbox("Dependents", cat_opts["Dependents"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)         # how many months the customer has been with us

    with c2:
        st.markdown("**📞 Services**")
        phone = st.selectbox("Phone Service", cat_opts["PhoneService"])
        multi = st.selectbox("Multiple Lines", cat_opts["MultipleLines"])
        internet = st.selectbox("Internet Service", cat_opts["InternetService"], index=1)
        security = st.selectbox("Online Security", cat_opts["OnlineSecurity"])
        backup = st.selectbox("Online Backup", cat_opts["OnlineBackup"])
        protection = st.selectbox("Device Protection", cat_opts["DeviceProtection"])

    with c3:
        st.markdown("**💳 Support, Streaming & Billing**")
        support = st.selectbox("Tech Support", cat_opts["TechSupport"])
        tv = st.selectbox("Streaming TV", cat_opts["StreamingTV"])
        movies = st.selectbox("Streaming Movies", cat_opts["StreamingMovies"])
        contract = st.selectbox("Contract", cat_opts["Contract"])
        paperless = st.selectbox("Paperless Billing", cat_opts["PaperlessBilling"])
        payment = st.selectbox("Payment Method", cat_opts["PaymentMethod"])

    m1, m2 = st.columns(2)
    monthly = m1.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total = m2.number_input("Total Charges ($)", 0.0, 10000.0, float(round(monthly * max(tenure, 1), 2)), step=10.0)

    if st.button("Predict Churn", type="primary"):
        # The new customer becomes a 1-row DataFrame. Column names must match training exactly.
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }])[meta["feature_columns"]]                         # column order same as training

        prob = float(pipe.predict_proba(row)[0, 1])          # probability of churn (0 to 1)
        label, icon = risk_label(prob)
        will_churn = prob >= THRESHOLD                       # threshold chosen in train.py

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Level", f"{icon} {label}")
        r3.metric("Prediction", "Will CHURN" if will_churn else "Will STAY")
        st.progress(min(prob, 1.0))                          # progress bar = visual of the probability

        if will_churn:
            st.warning("This customer is likely to churn. Try these retention actions:")
            for t in suggestion(row.iloc[0]):
                st.write("•", t)
        else:
            st.success("This customer looks safe for now. ✅")

# ===============================================================
# TAB 2: BATCH PREDICTION
# ===============================================================
with tab2:
    st.subheader("Upload a CSV and get the churn risk for all customers")
    st.caption("The CSV must have the same columns as the training data (customerID is optional, the Churn column is not needed).")

    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

        missing = [c for c in meta["feature_columns"] if c not in data.columns]
        if missing:
            st.error(f"These columns are missing: {missing}")
        else:
            X_new = data[meta["feature_columns"]]
            data["churn_probability"] = pipe.predict_proba(X_new)[:, 1].round(4)
            data["prediction"] = (data["churn_probability"] >= THRESHOLD).map({True: "Churn", False: "Stay"})
            data = data.sort_values("churn_probability", ascending=False)

            a, b = st.columns(2)
            a.metric("Total customers", len(data))
            b.metric("Predicted to churn", int((data["prediction"] == "Churn").sum()))
            st.dataframe(data.head(200), width="stretch")
            st.download_button("Download results", data.to_csv(index=False).encode(),
                               "churn_predictions.csv", "text/csv")

# ===============================================================
# TAB 3: MODEL PERFORMANCE
# ===============================================================
with tab3:
    st.subheader("How well does the model perform?")
    st.write(f"**Final model:** {meta['best_model']}  |  **Threshold:** {THRESHOLD:.3f}")

    cols = st.columns(5)
    for col, (k, v) in zip(cols, meta["test_metrics"].items()):
        col.metric(k.upper(), f"{v:.3f}")

    st.markdown("**Model comparison (5-fold CV, train data):**")
    if Path("models/model_comparison.csv").exists():
        st.dataframe(pd.read_csv("models/model_comparison.csv"), width="stretch", hide_index=True)

    g1, g2, g3 = st.columns(3)
    for col, img, cap in [(g1, "models/confusion_matrix.png", "Confusion Matrix"),
                          (g2, "models/roc_curve.png", "ROC Curve"),
                          (g3, "models/feature_importance.png", "Feature Importance")]:
        if Path(img).exists():
            col.image(img, caption=cap)
