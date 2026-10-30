import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "Churn_Modelling_Featured.csv"
RESULTS_PATH = "model_comparison_results.csv"

# =====================================================
# LOAD MODEL AND DATA
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Make sure best_churn_model.joblib is in the project folder.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Make sure Churn_Modelling_Featured.csv is in the project folder.")
    st.stop()

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Could not load the project files: {e}")
    st.stop()

# Target and feature columns
target = "Exited"

if target not in df.columns:
    st.error("The dataset must contain the target column 'Exited'.")
    st.stop()

feature_columns = [col for col in df.columns if col != target]
sample = df[feature_columns]

# =====================================================
# SESSION STATE AND NAVIGATION
# =====================================================
PAGES = ["Home", "Predict Churn", "Model Performance", "About Project"]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

def go_to(page):
    st.session_state.nav_page = page

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FB;
}

[data-testid="stSidebar"] {
    background: #101D35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent;
    border-radius: 9px;
    padding: 9px 12px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #233653;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    color: white;
    margin: 12px 0 4px 0;
}

.sidebar-subtitle {
    color: #AAB9CE;
    font-size: 12px;
    margin-bottom: 30px;
}

.sidebar-footer {
    color: #AAB9CE;
    font-size: 11px;
    margin-top: 45px;
    border-top: 1px solid #34445D;
    padding-top: 15px;
}

.hero {
    background: linear-gradient(115deg, #10213D 0%, #1D4772 65%, #286B91 100%);
    padding: 42px 42px;
    border-radius: 20px;
    color: white;
    margin-bottom: 26px;
    min-height: 245px;
}

.hero-tag {
    color: #B8DDF5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    line-height: 1.2;
    margin: 15px 0;
}

.hero p {
    color: #D5E3F1;
    font-size: 15px;
    max-width: 650px;
    line-height: 1.7;
}

.section-title {
    color: #172B49;
    font-size: 23px;
    font-weight: 800;
    margin: 20px 0 14px 0;
}

.metric-card {
    background: white;
    border: 1px solid #E5EAF1;
    border-radius: 15px;
    padding: 21px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(20, 45, 75, 0.04);
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-value {
    color: #17345A;
    font-size: 27px;
    font-weight: 800;
}

.metric-note {
    color: #8190A5;
    font-size: 11px;
    margin-top: 5px;
}

.info-card {
    background: white;
    border: 1px solid #E5EAF1;
    border-radius: 15px;
    padding: 23px;
    min-height: 145px;
    box-shadow: 0 3px 12px rgba(20, 45, 75, 0.04);
}

.info-card h3 {
    color: #18365C;
    font-size: 17px;
    margin-bottom: 9px;
}

.info-card p {
    color: #718096;
    font-size: 13px;
    line-height: 1.6;
}

div.stButton > button {
    background: #1D4F7A;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 11px 18px;
    font-weight: 700;
    min-height: 43px;
}

div.stButton > button:hover {
    background: #286B9B;
    color: white;
    border: none;
}

.result-box {
    background: white;
    border: 1px solid #E2E9F1;
    border-radius: 16px;
    padding: 25px;
    margin-top: 15px;
}

.small-muted {
    color: #78879A;
    font-size: 12px;
}

h1, h2, h3 {
    color: #172B49;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF1;
    padding: 16px;
    border-radius: 13px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏦 CHURN AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">BANK CUSTOMER INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("MAIN MENU")

    page = st.radio(
        "Navigation",
        PAGES,
        key="nav_page",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**PROJECT OVERVIEW**")
    st.markdown(
        f"<div class='small-muted'>Dataset records: {len(df):,}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='small-muted'>Input features: {len(feature_columns)}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-footer">Bank Customer Churn Prediction<br>AI & Machine Learning Project</div>',
        unsafe_allow_html=True
    )

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">AI POWERED CUSTOMER ANALYTICS</div>
        <h1>Bank Customer<br>Churn Prediction</h1>
        <p>
            Understand customer behavior, identify customers at risk of leaving,
            and support smarter business decisions using Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Customer Analysis</h3>
            <p>Explore customer data and identify patterns that may be related to customer churn.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Machine Learning</h3>
            <p>Use a trained Random Forest model to estimate the possibility of customer churn.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Business Insights</h3>
            <p>Turn model predictions into useful information that can support customer retention.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explore the System</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="info-card">
            <h3>Customer Prediction</h3>
            <p>Enter customer information and let the trained model predict the churn outcome.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Start a Customer Prediction →",
            use_container_width=True,
            on_click=go_to,
            args=("Predict Churn",)
        )

    with right:
        st.markdown("""
        <div class="info-card">
            <h3>Model Performance</h3>
            <p>Review the evaluation metrics and understand how the selected model performs.</p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Model Performance →",
            use_container_width=True,
            on_click=go_to,
            args=("Model Performance",)
        )

    st.markdown("---")
    st.caption("Bank Customer Churn Prediction | Machine Learning Deployment")

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "Predict Churn":

    st.title("Customer Churn Prediction")
    st.markdown(
        "Enter the customer's information below to generate a prediction."
    )
    st.markdown("---")

    st.markdown("### Customer Information")

    values = {}

    # Build input fields from the actual model dataset columns
    columns = st.columns(2)

    for i, col in enumerate(feature_columns):
        with columns[i % 2]:
            series = sample[col].dropna()

            if series.empty:
                values[col] = 0
                continue

            # Text / categorical columns
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or str(series.dtype) == "category"
            ):
                options = sorted(series.astype(str).unique().tolist())

                if len(options) <= 30:
                    values[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options,
                        key=f"input_{col}"
                    )
                else:
                    values[col] = st.text_input(
                        col.replace("_", " ").title(),
                        str(series.mode().iloc[0]),
                        key=f"input_{col}"
                    )

            # Numeric columns with a small number of unique values
            elif series.nunique() <= 10:
                options = sorted(series.unique().tolist())

                values[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options,
                    key=f"input_{col}"
                )

            # Other numeric columns
            else:
                default = float(series.median())
                minimum = float(series.min())
                maximum = float(series.max())

                if pd.api.types.is_integer_dtype(series):
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1,
                        key=f"input_{col}"
                    )
                else:
                    values[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        key=f"input_{col}"
                    )

    st.markdown("---")

    if st.button("Predict Customer Churn", use_container_width=True):

        try:
            input_df = pd.DataFrame([values], columns=feature_columns)

            # Match the original training column order
            prediction = model.predict(input_df)[0]

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### Prediction Result")

            if int(prediction) == 1:
                st.error("⚠️ This customer is predicted to leave the bank.")
                st.markdown("**Prediction: Churn**")
                st.write(
                    "The model identifies this customer as belonging to the churn class."
                )
            else:
                st.success("✅ This customer is predicted to stay with the bank.")
                st.markdown("**Prediction: Stay**")
                st.write(
                    "The model identifies this customer as belonging to the retained customer class."
                )

            # Show probability when supported by the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    churn_probability = float(probabilities[classes.index(1)]) * 100
                    st.metric("Churn Probability", f"{churn_probability:.1f}%")
                    st.progress(churn_probability / 100)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Please check that the model and dataset use the same features "
                "and column order as the original training process."
            )

# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================
elif page == "Model Performance":

    st.title("Model Performance")
    st.markdown(
        "Evaluation results of the selected Random Forest model."
    )
    st.markdown("---")

    # Reported test metrics from the project evaluation
    metrics = [
        ("Accuracy", "85.40%", "Overall correct predictions"),
        ("Precision", "74.89%", "Correct churn predictions"),
        ("Recall", "42.51%", "Churn customers identified"),
        ("F1-Score", "54.23%", "Balance between precision and recall"),
        ("ROC-AUC", "85.24%", "Ability to distinguish the classes")
    ]

    metric_cols = st.columns(5)

    for col, item in zip(metric_cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{item[0]}</div>
                <div class="metric-value">{item[1]}</div>
                <div class="metric-note">{item[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    cm = pd.DataFrame(
        [[1535, 58], [234, 173]],
        index=["Actual: Stayed", "Actual: Churned"],
        columns=["Predicted: Stayed", "Predicted: Churned"]
    )

    st.dataframe(cm, use_container_width=True)

    st.caption(
        "The matrix shows the number of correct and incorrect predictions "
        "for each customer class."
    )

    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Selected Model: Random Forest</h3>
        <p>
            Random Forest was selected based on its highest mean F1-score
            during Stratified 5-Fold Cross-Validation among the evaluated models.
        </p>
        <p>
            Cross-validation F1-score: <b>54.04%</b><br>
            Cross-validation ROC-AUC: <b>84.40%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Comparison Results</div>', unsafe_allow_html=True)

    if os.path.exists(RESULTS_PATH):
        try:
            results = pd.read_csv(RESULTS_PATH)
            st.dataframe(results, use_container_width=True)

            numeric_cols = results.select_dtypes(include=np.number).columns.tolist()

            if numeric_cols:
                selected_metric = st.selectbox(
                    "Choose a metric to visualize",
                    numeric_cols
                )

                st.bar_chart(
                    results.set_index(results.columns[0])[[selected_metric]]
                )

        except Exception as e:
            st.warning(f"Could not read model comparison results: {e}")
    else:
        st.info(
            "The model comparison file is not available. "
            "Add model_comparison_results.csv to display the comparison chart."
        )

# =====================================================
# ABOUT PROJECT PAGE
# =====================================================
elif page == "About Project":

    st.title("About the Project")
    st.markdown(
        "An end-to-end Machine Learning project for predicting bank customer churn."
    )
    st.markdown("---")

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">PROJECT DESCRIPTION</div>
        <h1>Turning Customer Data<br>into Predictions</h1>
        <p>
            This project aims to predict whether a bank customer is likely
            to leave the bank using historical customer information and
            Machine Learning techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Workflow</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Data Exploration", "Understand the dataset and explore its features."),
        ("02", "Data Preprocessing", "Handle missing values, duplicates, and data quality issues."),
        ("03", "Feature Engineering", "Prepare the features required for model training."),
        ("04", "Modeling", "Train and evaluate multiple Machine Learning algorithms."),
        ("05", "Model Selection", "Select Random Forest based on cross-validation F1-score."),
        ("06", "Deployment", "Deploy the trained model through a Streamlit web application.")
    ]

    for number, title, description in steps:
        st.markdown(f"""
        <div class="info-card" style="min-height:90px; margin-bottom:12px;">
            <h3>{number} · {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Technologies Used</div>', unsafe_allow_html=True)

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        <div class="info-card">
            <h3>Python</h3>
            <p>Main programming language used for data processing and model development.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech2:
        st.markdown("""
        <div class="info-card">
            <h3>Scikit-learn</h3>
            <p>Used to build, train, and evaluate Machine Learning models.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech3:
        st.markdown("""
        <div class="info-card">
            <h3>Streamlit</h3>
            <p>Used to create and deploy the interactive web application.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Bank Customer Churn Prediction | AI & Machine Learning Project")
