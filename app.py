"""
Telco Customer Churn Predictor (Streamlit app)

Run locally with:   streamlit run app.py
Needs these next to this file:
    features.py                  (feature engineering, created by the notebook)
    models/churn_model.joblib    (trained model, created by the notebook)
    data/sample_customers.csv    (optional demo file)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Importing prepare is required: the saved model refers to it, so Python must be able to find it.
from features import CAT_COLS, REQUIRED_COLUMNS, prepare  # noqa: F401

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
SAMPLE_PATH = BASE_DIR / "data" / "sample_customers.csv"

# The raw columns the model needs (defined in features.py so the app and model never disagree)
RAW_COLUMNS = REQUIRED_COLUMNS

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------- model helpers
@st.cache_resource
def load_bundle():
    """Load the saved model once and reuse it. Returns None if the file is missing."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_explainer(_pipeline):
    """SHAP explainer for the final model step (works for tree models such as XGBoost)."""
    try:
        return shap.TreeExplainer(_pipeline.named_steps["clf"])
    except Exception:
        return None


def score(df, bundle):
    """Add churn probability, risk band and retention flag to raw customer rows."""
    probs = bundle["pipeline"].predict_proba(df[RAW_COLUMNS])[:, 1]
    out = df.copy()
    out["churn_probability"] = probs.round(3)
    out["risk"] = pd.cut(probs, [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"],
                         include_lowest=True).astype(str)
    out["flag_for_retention"] = probs >= bundle["threshold"]
    return out


def friendly(name):
    """Turn 'cat__Contract_Two year' into 'Contract = Two year' for display."""
    name = name.split("__", 1)[-1]
    for col in CAT_COLS:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name.replace("_", " ")


def explain(row_df, bundle, top_n=8):
    """Top SHAP drivers for ONE customer (positive = pushes churn risk up)."""
    pipe = bundle["pipeline"]
    explainer = get_explainer(pipe)
    if explainer is None:
        return None
    pre = pipe.named_steps["pre"]
    X = pre.transform(pipe.named_steps["prep"].transform(row_df[RAW_COLUMNS].head(1)))
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    names = [friendly(n) for n in pre.get_feature_names_out()]
    s = pd.Series(values[0], index=names)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)


def plot_reasons(reasons):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    data = reasons.iloc[::-1]
    ax.barh(data.index, data.values, color=["#d62728" if v > 0 else "#1f77b4" for v in data.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Effect on churn risk  (red = raises risk, blue = lowers risk)")
    plt.tight_layout()
    return fig


def retention_ideas(c):
    """Simple rule-based suggestions. These are starting ideas, not model output."""
    ideas = []
    if c["Contract"] == "Month-to-month":
        ideas.append("Offer a discount to move to a 1-year or 2-year contract.")
    if c["InternetService"] == "Fiber optic" and c["TechSupport"] == "No":
        ideas.append("Offer a free tech-support trial (fiber customers without support churn more).")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        ideas.append("Offer a free online-security add-on trial to increase stickiness.")
    if c["PaymentMethod"] == "Electronic check":
        ideas.append("Give a small bill credit for switching to automatic payment.")
    if c["tenure"] <= 12:
        ideas.append("New customer: schedule an onboarding / check-in call in the first year.")
    if not ideas:
        ideas.append("Low-touch loyalty message or small thank-you reward.")
    return ideas


def show_customer_report(row_df, bundle):
    """Probability, risk, suggestions and the 'why' chart for the first row of row_df."""
    scored = score(row_df, bundle).iloc[0]
    prob = float(scored["churn_probability"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{prob:.0%}")
    c2.metric("Risk level", scored["risk"])
    c3.metric("Contact for retention?", "Yes" if scored["flag_for_retention"] else "No")
    st.progress(min(max(prob, 0.0), 1.0))

    left, right = st.columns(2)
    with left:
        st.subheader("Suggested retention actions")
        for idea in retention_ideas(row_df.iloc[0]):
            st.write(f"- {idea}")
    with right:
        st.subheader("Why this score?")
        try:
            reasons = explain(row_df, bundle)
        except Exception:
            reasons = None
        if reasons is None:
            st.info("Explanations are not available for this model type.")
        else:
            st.pyplot(plot_reasons(reasons))


# ---------------------------------------------------------------- page
st.title("📉 Telco Customer Churn Predictor")
st.caption("Predict which subscribers are likely to cancel, see why, and decide who to contact. "
           "Demo built on the public IBM Telco sample dataset (fictional customers).")

bundle = load_bundle()
if bundle is None:
    st.error("Model file not found: `models/churn_model.joblib`.")
    st.markdown("Run your notebook from top to bottom so it saves the model, then make sure the "
                "`models` folder sits next to `app.py`. See the deployment guide for details.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    st.metric("Retention threshold", f"{bundle['threshold']:.2f}")
    st.caption("Customers with a probability at or above this value are flagged for a retention offer. "
               "It was chosen in the notebook from the cost assumptions below.")
    a = bundle.get("assumptions", {})
    if a:
        st.write(f"- Offer cost: {a.get('cost_offer')}")
        st.write(f"- Value of a saved customer: {a.get('value_saved')}")
        st.write(f"- Save rate: {a.get('save_rate')}")

tab_single, tab_batch, tab_about = st.tabs(["Single customer", "Batch upload (CSV)", "About"])

# ---------------------------------------------------------------- single customer
with tab_single:
    st.subheader("Describe one customer")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profile**")
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months with company)", 0, 72, 12)
        with c2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            lines = st.selectbox("Multiple lines", ["No", "Yes"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", ["No", "Yes"])
            tech = st.selectbox("Tech support", ["No", "Yes"])
        with c3:
            st.markdown("**Contract and billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges", 0.0, 500.0, 70.0, step=1.0)
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        customer = pd.DataFrame([{
            "SeniorCitizen": 1 if senior == "Yes" else 0, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
            "OnlineSecurity": security, "TechSupport": tech, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": round(tenure * monthly, 2),   # estimated as tenure x monthly charges
        }])
        st.caption("Total charges are estimated as tenure x monthly charges.")
        show_customer_report(customer, bundle)

# ---------------------------------------------------------------- batch upload
with tab_batch:
    st.subheader("Score many customers at once")
    st.write("Upload a CSV in the style of the original Telco file. It needs at least the columns "
             "the model uses (listed in the error message if any are missing). Extra columns such as "
             "`customerID`, `gender` or `Churn` are fine and are ignored by the model.")

    if SAMPLE_PATH.exists():
        st.download_button("Download a sample CSV", SAMPLE_PATH.read_bytes(),
                           file_name="sample_customers.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    use_sample = st.checkbox("Or just try the sample file", value=False, disabled=not SAMPLE_PATH.exists())

    raw = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    elif use_sample:
        raw = pd.read_csv(SAMPLE_PATH)

    if raw is not None:
        raw.columns = [str(c).strip() for c in raw.columns]
        missing = [c for c in RAW_COLUMNS if c not in raw.columns]
        if missing:
            st.error("Your file is missing these columns: " + ", ".join(missing))
        else:
            scored = score(raw, bundle)
            id_col = ["customerID"] if "customerID" in scored.columns else []
            show_cols = id_col + ["tenure", "Contract", "InternetService", "MonthlyCharges",
                                  "churn_probability", "risk", "flag_for_retention"]
            ranked = scored.sort_values("churn_probability", ascending=False)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers", f"{len(scored):,}")
            m2.metric("Flagged for retention", f"{int(scored['flag_for_retention'].sum()):,}")
            m3.metric("High risk", f"{int((scored['risk'] == 'High').sum()):,}")
            m4.metric("Average probability", f"{scored['churn_probability'].mean():.0%}")

            st.bar_chart(scored["risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0))
            st.dataframe(
                ranked[show_cols], hide_index=True,
                column_config={"churn_probability": st.column_config.ProgressColumn(
                    "churn_probability", min_value=0.0, max_value=1.0, format="%.2f")})
            st.download_button("Download results as CSV", ranked.to_csv(index=False).encode("utf-8"),
                               file_name="churn_scores.csv", mime="text/csv")

            st.divider()
            st.subheader("Explain one of the riskiest customers")
            top = ranked.head(25)
            labels = {i: f"{top.loc[i, 'customerID'] if id_col else 'Row ' + str(i)}  "
                         f"(risk {top.loc[i, 'churn_probability']:.0%})" for i in top.index}
            chosen = st.selectbox("Customer", list(labels), format_func=lambda i: labels[i])
            show_customer_report(raw.loc[[chosen]], bundle)

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("How to read the results")
    st.markdown(
        """
- **Churn probability** is the model's score for how likely the customer is to cancel. Use it to
  *rank* customers. Because the model was trained to catch churners, the percentage is not a literal
  real-world probability.
- **Risk level** is a simple band: Low (under 30%), Medium (30-60%), High (over 60%).
- **Contact for retention?** uses the cost-based threshold from the sidebar, which balances the cost
  of an offer against the value of keeping a customer.
- **Why this score?** shows the factors that pushed the score up (red) or down (blue) for that customer.
- **Retention actions** are rule-based ideas to start from, not model output.

**Limits:** trained on a small public sample dataset of fictional customers, so this is a portfolio
demo, not something to use for real business decisions.
        """
    )
