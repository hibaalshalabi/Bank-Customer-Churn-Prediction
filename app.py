import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Bank Customer Churn Prediction | Streamlit Dashboard
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "Churn_Modelling_Featured.csv"

# Visual styling with fixed contrast for clear text reading
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f4f7fb; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071d3b 0%, #0b2b52 100%);
    }
    [data-testid="stSidebar"] * { color: #f3f7ff !important; }
    .brand { font-size: 1.25rem; font-weight: 800; line-height: 1.25; color: white; }
    .brand-sub { color: #b7c9e3 !important; font-size: .82rem; margin-top: .45rem; }
    .hero {
        padding: 2.1rem 2.2rem; border-radius: 22px;
        background: linear-gradient(115deg, #09264b 0%, #12558a 65%, #167b91 100%);
        color: white; margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(9,38,75,.15);
    }
    .hero h1 { color: white !important; font-size: 2.25rem; margin: .25rem 0 .6rem 0; }
    .hero p { color: #e1edf9 !important; font-size: 1rem; max-width: 800px; }
    .eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .75rem; font-weight: 700; color: #9edee5; }
    .section-title { color: #102d52; font-size: 1.35rem; font-weight: 800; margin: .5rem 0 1rem 0; }
    .soft-card {
        background: #ffffff; border: 1px solid #e2eaf4; border-radius: 17px;
        padding: 1.15rem 1.25rem; box-shadow: 0 5px 18px rgba(25,55,90,.045);
        color: #102d52;
    }
    .soft-card h3 { color: #102d52 !important; }
    .soft-card p, .soft-card li { color: #334e68 !important; }
    .metric-label { color: #61748d; font-size: .84rem; font-weight: 600; }
    .metric-value { color: #102d52; font-size: 1.75rem; font-weight: 800; margin-top: .2rem; }
    .muted { color: #486581 !important; }
    .result-good {
        border-radius: 18px; padding: 1.5rem; background: #e9faf4;
        border: 1px solid #b7ead8; color: #087b5b;
    }
    .result-risk {
        border-radius: 18px; padding: 1.5rem; background: #fff1ee;
        border: 1px solid #ffd1c8; color: #b53d2e;
    }
    .result-title { font-size: 1.55rem; font-weight: 800; margin: .35rem 0; }
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #1769e0, #1389a0);
        color: white; border: 0; border-radius: 10px; font-weight: 700;
        min-height: 2.8rem;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        color: white; border: 0; filter: brightness(1.05);
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e2eaf4; border-radius: 15px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetric"] label { color: #61748d !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #102d52 !important; }
    h2, h3 { color: #102d52; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Make sure it is in the app repository.")
    st.stop()
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file not found: `{DATA_PATH}`. Make sure it is in the app repository.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    data = load_data(DATA_PATH)
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

if "Exited" not in data.columns:
    st.error("The dataset must contain the target column `Exited`.")
    st.stop()

feature_data = data.drop(columns=["Exited"])
feature_columns = feature_data.columns.tolist()

# Initialize session state for navigation
pages = ["Home", "Predict Churn", "Model Performance", "About Project"]
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🏦 Bank Churn")
    st.markdown('<div class="brand">PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Customer retention through data-driven insights</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Use index to control selected page from session state reliably
    current_index = pages.index(st.session_state["current_page"])
    selected_page = st.radio(
        "NAVIGATION",
        pages,
        index=current_index,
        key="nav_radio",
        label_visibility="visible",
    )
    
    # Update session state if changed manually from sidebar
    if selected_page != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_page
        st.rerun()

    st.markdown("---")
    st.caption("Machine Learning • Random Forest")
    st.caption("A student project for customer churn analysis")

page = st.session_state["current_page"]

# Hero header & Content based on active page
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Customer intelligence platform</div>
        <h1>Bank Customer Churn Prediction</h1>
        <p>Understand customer churn risk with a machine-learning model. Explore a customer profile,
        estimate churn probability, and use the insights to support proactive retention decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project at a glance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Customer records", f"{len(data):,}")
    with c2:
        st.metric("Input features", f"{len(feature_columns)}")
    with c3:
        st.metric("Selected model", "Random Forest")
    with c4:
        st.metric("Test accuracy", "85.40%")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("""
        <div class="soft-card">
            <h3>🎯 What this application does</h3>
            <p class="muted">Enter customer information and receive a model-based prediction of whether the customer may leave the bank.</p>
            <ul>
              <li>Creates a prediction from the saved trained model.</li>
              <li>Displays the estimated probability of churn.</li>
              <li>Summarizes the model's evaluation metrics.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        # Fixed Start Button Action
        if st.button("Start a customer prediction →", use_container_width=True):
            st.session_state["current_page"] = "Predict Churn"
            st.rerun()
            
    with right:
        st.markdown("""
        <div class="soft-card">
            <h3>📌 Why churn prediction matters</h3>
            <p class="muted">Finding customers who may be at risk can help a bank prioritize follow-up and design customer-retention strategies.</p>
            <hr>
            <p><b>Important:</b> Predictions are estimates from a machine-learning model, not guarantees of future behavior.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("Use **Predict Churn** in the sidebar to enter customer details, or **Model Performance** to review evaluation results.")

elif page == "Predict Churn":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Customer risk assessment</div>
        <h1>Predict Customer Churn</h1>
        <p>Fill in the customer details below. The saved model will estimate the likelihood of churn.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Customer information</div>', unsafe_allow_html=True)
    customer_values = {}

    with st.form("customer_form", clear_on_submit=False):
        cols = st.columns(2)
        for index, column in enumerate(feature_columns):
            series = feature_data[column].dropna()
            if series.empty:
                st.error(f"No available values for `{column}`.")
                st.stop()

            with cols[index % 2]:
                is_categorical = (
                    pd.api.types.is_object_dtype(feature_data[column])
                    or isinstance(feature_data[column].dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(feature_data[column])
                )
                if is_categorical:
                    options = series.astype(str).unique().tolist()
                    customer_values[column] = st.selectbox(
                        column.replace("_", " "),
                        options=options,
                        key=f"input_{column}",
                    )
                elif pd.api.types.is_numeric_dtype(feature_data[column]):
                    unique_values = sorted(series.unique().tolist())
                    if len(unique_values) <= 2:
                        customer_values[column] = st.selectbox(
                            column.replace("_", " "),
                            options=unique_values,
                            format_func=lambda x: str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x),
                            key=f"input_{column}",
                        )
                    elif pd.api.types.is_integer_dtype(feature_data[column]):
                        customer_values[column] = st.number_input(
                            column.replace("_", " "),
                            min_value=int(series.min()),
                            max_value=int(series.max()),
                            value=int(series.median()),
                            step=1,
                            key=f"input_{column}",
                        )
                    else:
                        customer_values[column] = st.number_input(
                            column.replace("_", " "),
                            min_value=float(series.min()),
                            max_value=float(series.max()),
                            value=float(series.median()),
                            key=f"input_{column}",
                        )
                else:
                    options = series.astype(str).unique().tolist()
                    customer_values[column] = st.selectbox(
                        column.replace("_", " "),
                        options=options,
                        key=f"input_{column}",
                    )

        submitted = st.form_submit_button("✨  Predict Customer Churn", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([customer_values], columns=feature_columns)
        for column in feature_columns:
            if pd.api.types.is_numeric_dtype(feature_data[column]):
                input_df[column] = pd.to_numeric(input_df[column], errors="coerce")

        try:
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            churn_probability = float(probabilities[classes.index(1)])

            st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
            result_col, probability_col = st.columns([1.1, 1])
            with result_col:
                if prediction == 1:
                    st.markdown("""
                    <div class="result-risk">
                      <div>⚠️ HIGHER CHURN CONCERN</div>
                      <div class="result-title">Predicted to leave</div>
                      <div>The model classifies this customer as likely to churn.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-good">
                      <div>✓ RETENTION OUTLOOK</div>
                      <div class="result-title">Predicted to stay</div>
                      <div>The model classifies this customer as likely to remain with the bank.</div>
                    </div>
                    """, unsafe_allow_html=True)
            with probability_col:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.metric("Estimated churn probability", f"{churn_probability:.1%}")
                st.progress(churn_probability)
                st.caption("Probability estimated by the trained model.")
                st.markdown('</div>', unsafe_allow_html=True)
            st.caption("This is a model prediction, not a guarantee of the customer's future behavior.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info("Check that the model and dataset were created using the same feature columns.")

elif page == "Model Performance":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Evaluation and validation</div>
        <h1>Model Performance</h1>
        <p>Results from the final test set and cross-validation comparison of the trained classifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Random Forest — test set</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "85.40%")
    m2.metric("Precision", "74.89%")
    m3.metric("Recall", "42.51%")
    m4.metric("F1-score", "54.23%")
    m5.metric("ROC-AUC", "85.24%")

    st.caption("The model was selected using the highest mean F1-score across stratified 5-fold cross-validation.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="soft-card"><h3>Confusion matrix</h3><p class="muted">Actual vs. predicted labels on the 2,000-row test set.</p></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        matrix = [[1535, 58], [234, 173]]
        ax.imshow(matrix, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], labels=["Stayed (0)", "Churned (1)"])
        ax.set_yticks([0, 1], labels=["Stayed (0)", "Churned (1)"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown('<div class="soft-card"><h3>Cross-validation F1-score</h3><p class="muted">Mean F1-score for each candidate model.</p></div>', unsafe_allow_html=True)
        names = ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"]
        scores = [0.5404, 0.5085, 0.4989, 0.4867]
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        bars = ax.bar(names, scores)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel("Mean F1-score")
        ax.set_title("Model Comparison")
        ax.tick_params(axis="x", rotation=18)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown('<div class="section-title">Cross-validation summary</div>', unsafe_allow_html=True)
    results = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"],
        "Accuracy": [0.8521, 0.7289, 0.8342, 0.7924],
        "Precision": [0.7363, 0.4032, 0.6491, 0.4904],
        "Recall": [0.4270, 0.6890, 0.4055, 0.4834],
        "F1-score": [0.5404, 0.5085, 0.4989, 0.4867],
        "ROC-AUC": [0.8440, 0.7804, 0.7749, 0.6774],
    })
    st.dataframe(
        results.style.format({col: "{:.2%}" for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]}),
        use_container_width=True,
        hide_index=True,
    )
    st.warning("Recall for the churn class is 42.51%, so the model misses some customers who actually churn. This is an important limitation to discuss.")

else:
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Project documentation</div>
        <h1>About the Project</h1>
        <p>An end-to-end machine-learning project focused on identifying bank customers who may leave.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project workflow</div>', unsafe_allow_html=True)
    steps = st.columns(4)
    workflow = [
        ("01", "Data cleaning", "Handled missing values, duplicates, and data-quality issues."),
        ("02", "EDA & features", "Explored churn patterns and created useful features."),
        ("03", "Modeling", "Compared four classifiers using stratified cross-validation."),
        ("04", "Deployment", "Built an interactive Streamlit application for predictions."),
    ]
    for col, (number, title, description) in zip(steps, workflow):
        with col:
            st.markdown(
                f'<div class="soft-card"><div class="eyebrow">{number}</div><h3>{title}</h3><p class="muted">{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.markdown("""
    <div class="soft-card">
      <h3>🎯 Project objective</h3>
      <p class="muted">Predict whether a bank customer is likely to leave, helping inform customer-retention decisions.</p>
      <h3>🧰 Technologies</h3>
      <p class="muted">Python · Pandas · Scikit-learn · Joblib · Streamlit · Matplotlib</p>
      <h3>⚠️ Responsible use</h3>
      <p class="muted">The output is a statistical estimate and should be considered alongside human review and other relevant information.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#71839a;font-size:.8rem;padding:.4rem 0 1rem 0;">'
    'Bank Customer Churn Prediction · Machine Learning Project · Built with Streamlit</div>',
    unsafe_allow_html=True,
)
