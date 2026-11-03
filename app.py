"""
CUSTOMER CHURN PREDICTOR - STREAMLIT APP
=========================================
Run:  streamlit run app.py
(The model files must exist in the models/ folder. Run `python train.py` to recreate them.)

Streamlit's rule: whenever the user changes any widget, the whole script reruns top to bottom.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# PAGE SETTINGS (must be the very first streamlit command)
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------
# MODEL LOAD (cached: loaded only once, not on every click)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/churn_pipeline.joblib")   # preprocessing + model together
    with open("models/meta.json") as f:
        meta = json.load(f)                              # threshold, metrics, dropdown options
    return pipe, meta


if not Path("models/churn_pipeline.joblib").exists():
    st.error("Model file not found. Please run `python train.py` in the terminal first.")
    st.stop()

pipe, meta = load_artifacts()
THRESHOLD = meta["threshold"]
cat_opts = meta["categorical_options"]
FEATURES = meta["feature_columns"]
NUMERIC_COLS = list(meta.get("numeric_ranges", {}).keys()) or [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def risk_label(p):
    """Convert a probability into a business-friendly risk level."""
    if p >= 0.6:
        return "HIGH", "🔴"
    if p >= THRESHOLD:
        return "MEDIUM", "🟠"
    return "LOW", "🟢"


def suggestion(row):
    """Simple rule-based ideas to investigate (hand-written rules, NOT learned by the model)."""
    tips = []
    if row["Contract"] == "Month-to-month":
        tips.append("Month-to-month customers churn the most in this data: consider a discount for a 1- or 2-year contract.")
    if row["InternetService"] == "Fiber optic":
        tips.append("Fiber optic users churn more in this data: collect price/quality feedback and consider a loyalty offer.")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        tips.append("Consider a free/discounted Tech Support trial; customers with support churn less in this data.")
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        tips.append("Consider a free trial of the Online Security add-on.")
    if row["PaymentMethod"] == "Electronic check":
        tips.append("Consider a small cashback for switching to auto-pay (card/bank transfer).")
    if row["tenure"] < 12:
        tips.append("New customer (under 12 months): consider an onboarding call and a welcome offer.")
    if not tips:
        tips.append("No specific rule matched this customer. A general check-in or loyalty offer may still be worth considering.")
    return tips


def dependent_select(label, options, parent_is_off, off_value):
    """
    Some services only make sense if a parent service exists
    (for example, Tech Support needs Internet Service).
    If the parent is OFF, the only valid answer is `off_value` and the box is locked.
    If the parent is ON, `off_value` is removed from the choices.
    """
    if parent_is_off:
        return st.selectbox(label, [off_value], disabled=True)
    valid = [o for o in options if o != off_value]
    return st.selectbox(label, valid)


def load_and_validate_csv(uploaded):
    """Return (dataframe, None) if the file is usable, or (None, error_message) if not."""
    try:
        uploaded.seek(0)
        data = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        return None, "The file is empty. Please upload a CSV that contains customer data."
    except (pd.errors.ParserError, UnicodeDecodeError):
        return None, "Could not read this file as a CSV. Please check the file format."

    if data.empty:
        return None, "The file has column headers but no customer rows."

    data.columns = data.columns.str.strip()

    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        return None, f"These required columns are missing: {missing}"

    # Convert numeric columns. Anything that is not a number becomes NaN.
    for c in NUMERIC_COLS:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # TotalCharges may be blank for brand-new customers (same as training data), so it is allowed.
    # The other numeric columns must always have valid numbers.
    strict_cols = [c for c in NUMERIC_COLS if c != "TotalCharges" and c in FEATURES]
    bad_rows = data[strict_cols].isnull().any(axis=1)
    if bad_rows.any():
        line_numbers = (data.index[bad_rows] + 2).tolist()[:10]   # +2: header is line 1
        return None, (
            f"Some rows have missing or non-numeric values in {strict_cols}. "
            f"First problem lines in your file: {line_numbers}"
        )

    return data, None


# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption(
    "Estimates the probability that a telecom customer will leave (churn). "
    "This is a risk estimate to help prioritise retention work, not a certainty."
)

tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)", "📊 Model Performance"])

# ===============================================================
# TAB 1: SINGLE CUSTOMER
# ===============================================================
with tab1:
    st.subheader("Enter the customer's details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", cat_opts["gender"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", cat_opts["Partner"])
        dependents = st.selectbox("Dependents", cat_opts["Dependents"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        st.markdown("**📞 Services**")
        phone = st.selectbox("Phone Service", cat_opts["PhoneService"])
        multi = dependent_select("Multiple Lines", cat_opts["MultipleLines"],
                                 parent_is_off=(phone == "No"), off_value="No phone service")
        internet = st.selectbox("Internet Service", cat_opts["InternetService"], index=1)
        no_internet = (internet == "No")
        security = dependent_select("Online Security", cat_opts["OnlineSecurity"],
                                    no_internet, "No internet service")
        backup = dependent_select("Online Backup", cat_opts["OnlineBackup"],
                                  no_internet, "No internet service")
        protection = dependent_select("Device Protection", cat_opts["DeviceProtection"],
                                      no_internet, "No internet service")

    with c3:
        st.markdown("**💳 Support, Streaming & Billing**")
        support = dependent_select("Tech Support", cat_opts["TechSupport"],
                                   no_internet, "No internet service")
        tv = dependent_select("Streaming TV", cat_opts["StreamingTV"],
                              no_internet, "No internet service")
        movies = dependent_select("Streaming Movies", cat_opts["StreamingMovies"],
                                  no_internet, "No internet service")
        contract = st.selectbox("Contract", cat_opts["Contract"])
        paperless = st.selectbox("Paperless Billing", cat_opts["PaperlessBilling"])
        payment = st.selectbox("Payment Method", cat_opts["PaymentMethod"])

    m1, m2 = st.columns(2)
    monthly = m1.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total_input = m2.number_input(
        "Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0,
        disabled=(tenure == 0),
        help="Roughly Monthly Charges x Tenure. Locked when tenure is 0.",
    )
    if tenure == 0:
        m2.caption("Tenure is 0, so Total Charges is treated as missing (same as in the training data).")
        total = np.nan
    else:
        total = total_input

    if st.button("Predict Churn", type="primary"):
        # The customer becomes a 1-row DataFrame. Column names must match training exactly.
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
            "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }])[FEATURES]

        prob = float(pipe.predict_proba(row)[0, 1])      # probability of churn (0 to 1)
        label, icon = risk_label(prob)
        flagged = prob >= THRESHOLD

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Level", f"{icon} {label}")
        r3.metric("Decision", "Flagged" if flagged else "Not flagged")
        st.progress(min(prob, 1.0))
        st.caption(f"Decision rule: flagged when probability is at least {THRESHOLD:.3f}.")

        if flagged:
            st.warning("Flagged for retention review. Some ideas to investigate (hand-written rules, not proven fixes):")
            for t in suggestion(row.iloc[0]):
                st.write("•", t)
        else:
            st.success("Below the selected risk threshold. This does not guarantee the customer will stay.")

# ===============================================================
# TAB 2: BATCH PREDICTION
# ===============================================================
with tab2:
    st.subheader("Upload a CSV and get the churn risk for all customers")
    st.caption(
        "The CSV must contain the same feature columns as the training data. "
        "customerID is optional and the Churn column is not needed."
    )
    st.caption("Required columns: " + ", ".join(FEATURES))

    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        data, error = load_and_validate_csv(file)

        if error:
            st.error(error)
        else:
            # Warn (not stop) if a category was never seen in training.
            unknown = {}
            for col, allowed in cat_opts.items():
                if col in data.columns:
                    extra = set(data[col].dropna().astype(str).unique()) - set(allowed)
                    if extra:
                        unknown[col] = sorted(extra)
            if unknown:
                st.warning(f"Some values were not seen during training and will be ignored by the model: {unknown}")

            proba = None
            try:
                proba = pipe.predict_proba(data[FEATURES])[:, 1]
            except Exception as exc:
                st.error(f"Prediction failed. Please check your data. Details: {exc}")

            if proba is not None:
                results = data.copy()
                flagged_mask = proba >= THRESHOLD                  # compare the RAW probability first
                results["churn_probability"] = np.round(proba, 4)  # round only for display
                results["decision"] = np.where(flagged_mask, "Flagged for review", "Below threshold")
                results = results.sort_values("churn_probability", ascending=False)

                a, b = st.columns(2)
                a.metric("Total customers", len(results))
                b.metric("Flagged for retention review", int(flagged_mask.sum()))
                st.dataframe(results.head(200), width="stretch")
                if len(results) > 200:
                    st.caption("Showing the top 200 rows by risk. The download contains all rows.")
                st.download_button("Download results", results.to_csv(index=False).encode(),
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
