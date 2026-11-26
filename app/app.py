"""
app.py - Bank Marketing Prediction System (Streamlit user interface)
=====================================================================
Run from the project root:
    streamlit run app/app.py

How it works
------------
1. The trained Pipeline (preprocessing + model) is loaded ONCE from
   models/final_model.joblib with joblib (cached by Streamlit).
2. The user fills in a form with the client's information.
3. When "Predict" is pressed, src/predict.py validates the inputs and calls
   model.predict_proba() on them - the SAME code used in notebook 05.
4. The probability is compared with the decision threshold chosen in
   notebook 04, and the app shows YES or NO.
Nothing is random or hard-coded: every result comes from the real model.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Make the project's src/ package importable when running `streamlit run app/app.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from src.predict import load_model, predict_customer  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bank Marketing Prediction System", page_icon="🏦",
                   layout="wide")

st.markdown("""
<style>
    .hero {background: linear-gradient(120deg, #1c5cab 0%, #2a78d6 100%); color: white;
           padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin: 0 0 .3rem 0; font-size: 2.1rem;}
    .hero p {color: #e8f0fb; margin: 0; font-size: 1.05rem;}
    .result-yes, .result-no {padding: 1.3rem 1.6rem; border-radius: 14px; margin: .4rem 0 1rem 0;}
    .result-yes {background: #fdeee7; border-left: 8px solid #eb6834;}
    .result-no  {background: #eaf2fc; border-left: 8px solid #2a78d6;}
    .result-yes h2, .result-no h2 {margin: 0; color: #0b0b0b;}
    .result-yes p, .result-no p {margin: .3rem 0 0 0; color: #333;}
    .small-note {color: #52514e; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Load the model once and reuse it for every prediction (fast)."""
    return load_model()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🏦 Bank Marketing Prediction System</h1>
  <p>Predict whether a bank client will subscribe to a term deposit <b>before</b> calling them.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, meta = get_model()
except FileNotFoundError as err:
    st.error(f"**The trained model was not found.** {err}")
    st.stop()

test_results = read_json(RESULTS_DIR / "final_test_metrics.json")

# ---------------------------------------------------------------------------
# Sidebar: model information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About the model")
    st.write(f"**Algorithm:** {meta['model_name']}")
    st.write(f"**Decision threshold:** {meta['decision_threshold']:.3f}")
    st.caption("A client is predicted **YES** when the predicted probability of "
               "subscribing is at least this threshold. It was chosen on training "
               "data to balance precision and recall (max F1).")
    if test_results:
        m = test_results["test_metrics"]
        st.subheader("Test-set performance")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        c2.metric("F1-score", f"{m['f1']:.3f}")
        c1.metric("Recall", f"{m['recall']:.3f}")
        c2.metric("Precision", f"{m['precision']:.3f}")
    st.divider()
    st.caption("Data: Moro, Rita & Cortez (2014), UCI Bank Marketing dataset (ID 222). "
               "The call `duration` is NOT used because it is only known after the call.")

tab_predict, tab_about, tab_perf = st.tabs(
    ["🔮 Predict a client", "📘 About the project", "📊 Model performance"])

# ---------------------------------------------------------------------------
# Tab 1: Prediction form
# ---------------------------------------------------------------------------
with tab_predict:
    cats = meta["categories"]
    ranges = meta["numeric_ranges"]          # [min, max, median] from training data
    NOT_GIVEN = "— not provided —"

    def choice(label, column, help_text=None, default=None):
        """Select box limited to the categories seen during training.
        Choosing 'not provided' sends a missing value (handled by the imputer)."""
        options = [NOT_GIVEN] + cats[column]
        index = options.index(default) if default in options else 0
        value = st.selectbox(label, options, index=index, help=help_text)
        return None if value == NOT_GIVEN else value

    st.subheader("Client information")
    st.markdown('<p class="small-note">Fill in what you know. Empty fields are allowed: a missing '
                'number is replaced by the training median and a missing category by "unknown".</p>',
                unsafe_allow_html=True)

    # Outside the form so that the campaign fields below react immediately.
    contacted_before = st.toggle("This client was contacted in a PREVIOUS campaign", value=False)

    with st.form("client_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Personal")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1)
            job = choice("Job", "job", default="management")
            marital = choice("Marital status", "marital", default="married")
            education = choice("Education", "education", default="secondary")

        with col2:
            st.markdown("##### 💶 Financial")
            balance = st.number_input("Average yearly balance (€)", value=ranges["balance"][2], step=100,
                                      help="Can be negative (overdraft).")
            default = choice("Has credit in default?", "default", default="no")
            housing = choice("Has a housing loan?", "housing", default="yes")
            loan = choice("Has a personal loan?", "loan", default="no")

        with col3:
            st.markdown("##### 📞 Campaign")
            contact = choice("Contact type", "contact", default="cellular")
            month = choice("Planned contact month", "month", default="may")
            day = st.number_input("Planned contact day of month", min_value=1, max_value=31, value=15)
            campaign = st.number_input("Number of contacts in this campaign (incl. this one)",
                                       min_value=1, max_value=100, value=1)
            if contacted_before:
                pdays = st.number_input("Days since last contact in a previous campaign",
                                        min_value=0, max_value=1000, value=180)
                previous = st.number_input("Number of contacts before this campaign",
                                           min_value=1, max_value=300, value=1)
                outcome_options = [o for o in cats["poutcome"] if o != "unknown"]
                poutcome = st.selectbox("Outcome of the previous campaign", outcome_options)
            else:
                # Encoding used in the original data for never-contacted clients
                pdays, previous, poutcome = -1, 0, "unknown"
                st.info("Never contacted before → pdays = -1, previous = 0, outcome = unknown")

        submitted = st.form_submit_button("🔮 Predict subscription", type="primary",
                                          width="stretch")

    if submitted:
        customer = {"age": age, "job": job, "marital": marital, "education": education,
                    "default": default, "balance": balance, "housing": housing, "loan": loan,
                    "contact": contact, "day": day, "month": month, "campaign": campaign,
                    "pdays": pdays, "previous": previous, "poutcome": poutcome}
        try:
            result = predict_customer(customer, model, meta)
        except Exception as err:  # never show a raw traceback to the user
            st.error(f"Prediction failed because of an unexpected error: {err}")
            st.stop()

        if not result["ok"]:
            st.error("**Please correct the following inputs:**\n\n" +
                     "\n".join(f"- {e}" for e in result["errors"]))
        else:
            prob, thr = result["probability_yes"], result["threshold"]
            if result["prediction"] == "YES":
                st.markdown(f"""<div class="result-yes"><h2>✅ YES — likely to subscribe</h2>
                    <p>Predicted probability of subscribing: <b>{prob:.1%}</b> (threshold {thr:.1%}).
                    This client is a good candidate to call.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-no"><h2>❌ NO — unlikely to subscribe</h2>
                    <p>Predicted probability of subscribing: <b>{prob:.1%}</b> (threshold {thr:.1%}).
                    Calling this client has a low expected success rate.</p></div>""",
                            unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Probability of YES", f"{prob:.1%}")
            c2.metric("Decision threshold", f"{thr:.1%}")
            c3.metric("Average client (base rate)", "11.7%",
                      delta=f"{(prob - 0.117) * 100:+.1f} pp vs average", delta_color="off")
            st.progress(min(max(prob, 0.0), 1.0), text="Predicted probability of subscribing")
            st.caption("The probability is a model estimate based on historical campaigns (2008–2010). "
                       "It supports — but does not replace — human judgement.")
            for w in result["warnings"]:
                st.warning(w)

# ---------------------------------------------------------------------------
# Tab 2: Project introduction
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("The prediction problem")
    st.markdown("""
A Portuguese bank ran telephone marketing campaigns to sell **term deposits**.
Only about **11.7 %** of the contacted clients subscribed, so most calls were unsuccessful.

**Goal:** a binary **classification** model that predicts, *before the call*, whether a client will
subscribe (**YES**) or not (**NO**), so that the bank can call the most promising clients first.

| | |
|---|---|
| **Data** | UCI Bank Marketing dataset (`bank-full.csv`): 45,211 clients, 16 input features |
| **Target** | `y` – did the client subscribe to a term deposit? (yes / no) |
| **Inputs used** | age, job, marital status, education, default, balance, housing loan, personal loan, contact type, day, month, number of contacts, days since previous contact, previous contacts, previous outcome |
| **Excluded** | `duration` of the call — it is only known *after* the call, so using it would be data leakage |
| **Models compared** | Dummy baseline, Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting |
| **Main metric** | ROC-AUC (ranking quality, robust to class imbalance) |
""")
    st.info("Source: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5K306")

# ---------------------------------------------------------------------------
# Tab 3: Model performance (real results produced by notebooks 03-05)
# ---------------------------------------------------------------------------
with tab_perf:
    if test_results is None:
        st.warning("Run notebook 05 to generate results/final_test_metrics.json.")
    else:
        st.subheader(f"Final model: {meta['model_name']}")
        m = test_results["test_metrics"]
        cols = st.columns(6)
        for col, key, label in zip(cols, ["roc_auc", "average_precision", "f1", "precision", "recall",
                                          "accuracy"],
                                   ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy"]):
            col.metric(label, f"{m[key]:.3f}")
        st.caption(f"Single evaluation on {test_results['n_test']:,} untouched test clients.")
        fig_cols = st.columns(2)
        for col, name in zip(fig_cols * 2, ["fig10_confusion_matrix.png", "fig11_roc_curve.png",
                                            "fig08_all_models_cv.png", "fig12_precision_recall_curve.png"]):
            if (FIGURES_DIR / name).exists():
                col.image(str(FIGURES_DIR / name), width="stretch")
