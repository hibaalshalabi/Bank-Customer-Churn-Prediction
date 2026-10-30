import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# BANK CUSTOMER CHURN PREDICTION | PRESENTATION-READY DASHBOARD
# ============================================================
st.set_page_config(
    page_title="Bank Customer Intelligence | Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "Churn_Modelling_Featured.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --navy:#10243e; --blue:#2563eb; --teal:#0f9f9a; --ink:#172b45; --muted:#687b91; --line:#e5ebf2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f5f7fb; color:var(--ink); }
.block-container { padding-top:2rem; padding-bottom:2rem; max-width:1440px; }
[data-testid="stSidebar"] { background:#10243e; border-right:1px solid #1d3858; }
[data-testid="stSidebar"] * { color:#edf4ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-weight:600; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:.45rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size:.95rem; }
.brand-mark { width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,#3b82f6,#14b8a6);display:flex;align-items:center;justify-content:center;font-size:23px;margin-bottom:15px; }
.sidebar-kicker { color:#91a9c7 !important;font-size:.68rem;letter-spacing:.16em;font-weight:800;text-transform:uppercase; }
.sidebar-name { color:#fff;font-family:Manrope,sans-serif;font-size:1.2rem;font-weight:800;line-height:1.25;margin:.2rem 0; }
.sidebar-desc { color:#a9bdd5 !important;font-size:.78rem;line-height:1.5;margin-bottom:1.3rem; }
.sidebar-foot { color:#8fa6c3 !important;font-size:.72rem;line-height:1.6; }
.topline { display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;color:#6b7e94;font-size:.82rem; }
.status-pill { display:inline-block;padding:.35rem .7rem;border-radius:99px;background:#e7f7f2;color:#087e68;font-weight:700;font-size:.72rem; }
.hero { background:linear-gradient(112deg,#10243e 0%,#174b75 58%,#087f8c 100%);border-radius:24px;padding:2.25rem 2.4rem;color:white;box-shadow:0 16px 38px rgba(16,36,62,.13);margin-bottom:1.55rem;position:relative;overflow:hidden; }
.hero:after { content:'';position:absolute;width:250px;height:250px;border:1px solid rgba(255,255,255,.13);border-radius:50%;right:-65px;top:-100px;box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 70px rgba(255,255,255,.025); }
.hero-kicker { color:#9de6e0;font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;font-weight:800; }
.hero h1 { color:#fff;font-family:Manrope,sans-serif;font-size:2.2rem;font-weight:800;letter-spacing:-.04em;margin:.45rem 0 .55rem; }
.hero p { color:#dceaf7;font-size:.98rem;max-width:790px;line-height:1.65;margin:0; }
.section-head { display:flex;align-items:center;gap:.65rem;margin:1.5rem 0 .9rem;font-family:Manrope,sans-serif;font-size:1.12rem;font-weight:800;color:#172f4d; }
.section-head:before { content:'';width:5px;height:22px;border-radius:5px;background:linear-gradient(#2563eb,#0faaa2);display:inline-block; }
.card { background:#fff;border:1px solid #e7edf4;border-radius:18px;padding:1.25rem 1.35rem;box-shadow:0 5px 20px rgba(27,53,82,.045);height:100%; }
.card-label { color:#71839a;font-size:.75rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase; }
.card-value { color:#142d4a;font-family:Manrope,sans-serif;font-size:1.75rem;font-weight:800;margin-top:.35rem; }
.card-note { color:#8493a6;font-size:.75rem;margin-top:.2rem; }
.feature-title { color:#193653;font-family:Manrope,sans-serif;font-weight:800;font-size:1.02rem;margin:.2rem 0 .55rem; }
.feature-copy { color:#6b7e93;font-size:.88rem;line-height:1.65; }
div.stButton > button, div.stFormSubmitButton > button { background:linear-gradient(100deg,#2563eb,#0f9f9a);color:#fff;border:0;border-radius:11px;font-weight:800;min-height:2.85rem;box-shadow:0 5px 14px rgba(37,99,235,.16); }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { color:#fff;border:0;filter:brightness(1.06); }
[data-testid="stMetric"] { background:#fff;border:1px solid #e7edf4;border-radius:15px;padding:1rem 1.05rem;box-shadow:0 4px 15px rgba(27,53,82,.035); }
[data-testid="stMetricLabel"] { color:#71839a; }
[data-testid="stMetricValue"] { color:#183653;font-family:Manrope,sans-serif;font-weight:800; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius:10px; }
.stTextInput label,.stNumberInput label,.stSelectbox label { color:#435a73;font-weight:700; }
.result-card { padding:1.5rem;border-radius:18px;border:1px solid;min-height:180px; }
.result-risk { background:#fff4f1;border-color:#ffd7ce;color:#a63c2d; }
.result-safe { background:#edfaf5;border-color:#c6ecdd;color:#08765c; }
.result-kicker { font-size:.72rem;font-weight:800;letter-spacing:.12em; }
.result-heading { font-family:Manrope,sans-serif;font-size:1.55rem;font-weight:800;margin:.5rem 0; }
.result-detail { font-size:.88rem;line-height:1.55; }
.small-note { color:#7c8da1;font-size:.78rem;line-height:1.5; }
div[data-testid="stDataFrame"] { border:1px solid #e5ebf2;border-radius:12px;overflow:hidden; }
hr { border-color:#e5ebf2; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


# Load artifacts with clear diagnostics
missing = [name for name in (MODEL_PATH, DATA_PATH) if not os.path.exists(name)]
if missing:
    st.error("Required project file(s) not found: " + ", ".join(missing))
    st.info("Keep app.py, best_churn_model.joblib, and Churn_Modelling_Featured.csv in the repository root.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    data = load_data(DATA_PATH)
except Exception as error:
    st.error(f"The model or dataset could not be loaded: {error}")
    st.stop()

if "Exited" not in data.columns:
    st.error("The dataset must contain the target column 'Exited'.")
    st.stop()

feature_data = data.drop(columns=["Exited"])
feature_columns = feature_data.columns.tolist()

# Sidebar
if "page_choice" not in st.session_state:
    st.session_state.page_choice = "Overview"
with st.sidebar:
    st.markdown('<div class="brand-mark">🏦</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-kicker">Machine learning project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-name">Bank Customer<br>Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-desc">Churn prediction & customer retention insights</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Predict Churn", "Model Evaluation", "Project Details"], key="page_choice", label_visibility="visible")
    st.markdown("---")
    st.markdown('<div class="sidebar-foot">MODEL<br><b>Random Forest Classifier</b><br><br>DEPLOYMENT<br><b>Streamlit Web Application</b><br><br>PROJECT<br><b>End-to-end machine learning</b></div>', unsafe_allow_html=True)


def hero(kicker, title, description):
    st.markdown(f'<div class="hero"><div class="hero-kicker">{kicker}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)


def info_card(title, copy, icon):
    st.markdown(f'<div class="card"><div style="font-size:1.5rem;margin-bottom:.65rem">{icon}</div><div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>', unsafe_allow_html=True)


st.markdown('<div class="topline"><span>ANALYTICS WORKSPACE&nbsp; / &nbsp; BANK CUSTOMER CHURN</span><span class="status-pill">● MODEL READY</span></div>', unsafe_allow_html=True)

if page == "Overview":
    hero("Customer analytics platform", "Bank Customer Churn Prediction", "A machine-learning application that estimates customer churn risk and presents model evaluation results to support data-informed retention decisions.")
    section("Project snapshot")
    metric_cols = st.columns(4)
    metrics = [("CUSTOMER RECORDS", f"{len(data):,}", "Prepared dataset"), ("MODEL INPUTS", str(len(feature_columns)), "Features used for prediction"), ("SELECTED MODEL", "Random Forest", "Best mean cross-validation F1"), ("TEST ACCURACY", "85.40%", "Held-out test set")]
    for col, (label, value, note) in zip(metric_cols, metrics):
        with col:
            st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-note">{note}</div></div>', unsafe_allow_html=True)
    section("Explore the application")
    a, b, c = st.columns(3)
    with a:
        info_card("Customer prediction", "Enter a customer profile and receive a churn classification with an estimated probability.", "🎯")
    with b:
        info_card("Model evaluation", "Review test-set metrics, confusion matrix, and cross-validation model comparison.", "📊")
    with c:
        info_card("Project workflow", "Explore the main stages from data preparation through model training and deployment.", "⚙️")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="card"><div class="feature-title">From data to decision</div><div class="feature-copy">The application uses a trained Random Forest model to estimate whether a customer may leave the bank. The result can help prioritize further review, but should not be treated as a guarantee.</div></div>', unsafe_allow_html=True)
    with right:
        if st.button("Open prediction workspace  →", use_container_width=True):
            st.session_state.page_choice = "Predict Churn"
            st.rerun()

elif page == "Predict Churn":
    hero("Customer risk assessment", "Predict Customer Churn", "Provide the customer information below. The deployed model will return a prediction and an estimated churn probability.")
    section("Customer profile")
    st.markdown('<div class="small-note">Complete the fields using the available dataset ranges. The form is arranged in two columns for easier review.</div>', unsafe_allow_html=True)
    customer_values = {}
    with st.form("customer_prediction_form", clear_on_submit=False):
        cols = st.columns(2)
        for index, column in enumerate(feature_columns):
            series = feature_data[column].dropna()
            if series.empty:
                st.error(f"No available values for {column}.")
                st.stop()
            with cols[index % 2]:
                label = column.replace("_", " ").title()
                if (pd.api.types.is_object_dtype(feature_data[column]) or isinstance(feature_data[column].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(feature_data[column])):
                    options = series.astype(str).unique().tolist()
                    customer_values[column] = st.selectbox(label, options=options, key=f"profile_{column}")
                elif pd.api.types.is_numeric_dtype(feature_data[column]):
                    unique_values = sorted(series.unique().tolist())
                    if len(unique_values) <= 2:
                        customer_values[column] = st.selectbox(label, options=unique_values, key=f"profile_{column}")
                    elif pd.api.types.is_integer_dtype(feature_data[column]):
                        customer_values[column] = st.number_input(label, min_value=int(series.min()), max_value=int(series.max()), value=int(series.median()), step=1, key=f"profile_{column}")
                    else:
                        customer_values[column] = st.number_input(label, min_value=float(series.min()), max_value=float(series.max()), value=float(series.median()), key=f"profile_{column}")
                else:
                    options = series.astype(str).unique().tolist()
                    customer_values[column] = st.selectbox(label, options=options, key=f"profile_{column}")
        submitted = st.form_submit_button("✦  Generate churn prediction", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([customer_values], columns=feature_columns)
        for column in feature_columns:
            if pd.api.types.is_numeric_dtype(feature_data[column]):
                input_df[column] = pd.to_numeric(input_df[column], errors="coerce")
        try:
            prediction = int(model.predict(input_df)[0])
            if not hasattr(model, "predict_proba"):
                raise ValueError("The selected model does not provide probability estimates.")
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            positive_class = next((i for i, value in enumerate(classes) if str(value) == "1"), None)
            if positive_class is None:
                raise ValueError("Could not identify the churn class (1) in the trained model.")
            churn_probability = float(probabilities[positive_class])
            section("Prediction outcome")
            result_col, probability_col = st.columns([1.1, 1])
            with result_col:
                if prediction == 1:
                    st.markdown('<div class="result-card result-risk"><div class="result-kicker">CHURN RISK INDICATOR</div><div class="result-heading">Predicted to leave</div><div class="result-detail">The model classifies this customer in the churn class. Consider this result as a signal for further review.</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-card result-safe"><div class="result-kicker">CUSTOMER RETENTION INDICATOR</div><div class="result-heading">Predicted to stay</div><div class="result-detail">The model classifies this customer in the non-churn class. This prediction is an estimate, not a guarantee.</div></div>', unsafe_allow_html=True)
            with probability_col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric("Estimated probability of churn", f"{churn_probability:.1%}")
                st.progress(churn_probability)
                st.markdown('<div class="small-note">Probability produced by the trained classification model.</div></div>', unsafe_allow_html=True)
            st.caption("Important: This tool supports analysis and does not replace human judgment or confirm future customer behavior.")
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")
            st.info("Verify that the model and dataset use the same feature names, order, and data types.")

elif page == "Model Evaluation":
    hero("Model validation", "Model Evaluation", "A transparent summary of the selected model's performance on unseen test data and its cross-validation results.")
    section("Random Forest · final test results")
    metric_cols = st.columns(5)
    for col, (label, value) in zip(metric_cols, [("ACCURACY", "85.40%"), ("PRECISION", "74.89%"), ("RECALL", "42.51%"), ("F1-SCORE", "54.23%"), ("ROC-AUC", "85.24%")]):
        with col:
            st.metric(label, value)
    st.markdown('<div class="small-note">The Random Forest model was selected based on the highest mean F1-score across stratified 5-fold cross-validation.</div>', unsafe_allow_html=True)
    section("Performance diagnostics")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><div class="feature-title">Confusion matrix</div><div class="feature-copy">Actual versus predicted classes on the 2,000-customer test set.</div></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        matrix = [[1535, 58], [234, 173]]
        ax.imshow(matrix, cmap="Blues")
        ax.set_xlabel("Predicted class", labelpad=10)
        ax.set_ylabel("Actual class", labelpad=10)
        ax.set_xticks([0, 1], labels=["Stayed", "Churned"])
        ax.set_yticks([0, 1], labels=["Stayed", "Churned"])
        ax.set_title("Test-set classification results", pad=14, fontweight="bold")
        threshold = max(max(row) for row in matrix) / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center", color="white" if matrix[i][j] > threshold else "#17324d", fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown('<div class="card"><div class="feature-title">Cross-validation F1-score</div><div class="feature-copy">Mean F1-score across the four evaluated classifiers.</div></div>', unsafe_allow_html=True)
        names = ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"]
        scores = [0.5404, 0.5085, 0.4989, 0.4867]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(names[::-1], scores[::-1], color=["#9bb8d8", "#9bb8d8", "#9bb8d8", "#2563eb"])
        ax.set_xlim(0, 0.65)
        ax.set_xlabel("Mean F1-score")
        ax.set_title("Classifier comparison", pad=14, fontweight="bold")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    section("Cross-validation comparison")
    results = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"],
        "Accuracy": [0.8521, 0.7289, 0.8342, 0.7924],
        "Precision": [0.7363, 0.4032, 0.6491, 0.4904],
        "Recall": [0.4270, 0.6890, 0.4055, 0.4834],
        "F1-score": [0.5404, 0.5085, 0.4989, 0.4867],
        "ROC-AUC": [0.8440, 0.7804, 0.7749, 0.6774],
    })
    st.dataframe(results.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}), use_container_width=True, hide_index=True)
    st.warning("Model limitation: the churn-class recall is 42.51%. This means the model does not identify every customer who actually leaves. This limitation should be considered when interpreting results.")

else:
    hero("Project documentation", "About This Project", "An end-to-end machine-learning project that explores bank customer data, trains classification models, and deploys a prediction interface.")
    section("Project workflow")
    workflow = [
        ("01", "Data preparation", "Reviewed the raw dataset, handled missing values and duplicates, and prepared a clean dataset."),
        ("02", "Exploration & features", "Explored customer churn patterns and prepared features for model development."),
        ("03", "Model development", "Compared Logistic Regression, KNN, Decision Tree, and Random Forest using stratified cross-validation."),
        ("04", "Deployment", "Saved the selected model and integrated it into an interactive Streamlit application."),
    ]
    for start in range(0, len(workflow), 2):
        row = st.columns(2)
        for col, (number, title, description) in zip(row, workflow[start:start+2]):
            with col:
                st.markdown(f'<div class="card"><div class="card-label">PROJECT STAGE {number}</div><div class="feature-title" style="margin-top:.45rem">{title}</div><div class="feature-copy">{description}</div></div>', unsafe_allow_html=True)
    section("Technology stack")
    tech1, tech2 = st.columns(2)
    with tech1:
        st.markdown('<div class="card"><div class="feature-title">Development & machine learning</div><div class="feature-copy">Python · Pandas · Scikit-learn · Joblib · Matplotlib</div></div>', unsafe_allow_html=True)
    with tech2:
        st.markdown('<div class="card"><div class="feature-title">Deployment & interface</div><div class="feature-copy">Streamlit · GitHub · Streamlit Community Cloud</div></div>', unsafe_allow_html=True)
    section("Responsible interpretation")
    st.markdown('<div class="card"><div class="feature-copy">The prediction is a statistical estimate generated from historical data. It should be used as decision support only, alongside appropriate human review. Model performance may vary when applied to new data or different customer populations.</div></div>', unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#8392a5;font-size:.75rem;padding:.25rem 0 1rem">BANK CUSTOMER CHURN PREDICTION &nbsp;·&nbsp; MACHINE LEARNING PROJECT &nbsp;·&nbsp; STREAMLIT</div>', unsafe_allow_html=True)
