
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# BANK CUSTOMER CHURN PREDICTION
# Professional Streamlit Dashboard
# =========================================================

st.set_page_config(
    page_title="Bank Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "Churn_Modelling_Featured.csv"
COMPARISON_PATH = "model_comparison_results.csv"

# ---------------------- DESIGN ---------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --navy: #0b1f3a;
    --blue: #1769e0;
    --teal: #13a6a6;
    --text: #183451;
    --muted: #718198;
    --border: #e3ebf4;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f3f6fb;
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1f3a, #12365d);
    border-right: 1px solid #1c4168;
}

[data-testid="stSidebar"] * {
    color: #f1f6ff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 12px;
    border-radius: 10px;
    font-weight: 600;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1a4776;
}

.brand-icon {
    font-size: 36px;
    margin-bottom: 10px;
}

.brand-title {
    font-family: Manrope, sans-serif;
    font-size: 19px;
    font-weight: 800;
    line-height: 1.35;
    color: white;
}

.brand-subtitle {
    color: #a8bdd7;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 8px;
}

.side-label {
    color: #88a8cc;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 28px 0 10px;
}

/* Header */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
    color: #72849a;
    font-size: 12px;
}

.status {
    background: #e5f7ef;
    color: #08795e;
    padding: 7px 12px;
    border-radius: 30px;
    font-weight: 800;
}

/* Hero */
.hero {
    background: linear-gradient(110deg, #0b1f3a, #174b78 65%, #087f8b);
    padding: 38px 40px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 30px rgba(13,39,68,.13);
}

.hero-kicker {
    color: #8fe2df;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: white;
    font-family: Manrope, sans-serif;
    font-size: 34px;
    font-weight: 800;
    margin: 12px 0;
    line-height: 1.2;
}

.hero p {
    color: #dce9f6;
    font-size: 14px;
    max-width: 800px;
    line-height: 1.7;
    margin-bottom: 0;
}

/* Section titles */
.section-title {
    color: #183451;
    font-family: Manrope, sans-serif;
    font-size: 19px;
    font-weight: 800;
    margin: 27px 0 15px;
    border-left: 4px solid #1769e0;
    padding-left: 12px;
}

/* KPI cards */
.kpi {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 19px;
    min-height: 120px;
    box-shadow: 0 5px 18px rgba(23,48,78,.04);
}

.kpi-label {
    color: #718198;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.kpi-value {
    color: #12365d;
    font-family: Manrope, sans-serif;
    font-size: 25px;
    font-weight: 800;
    margin: 9px 0 4px;
}

.kpi-note {
    color: #8795a7;
    font-size: 11px;
}

/* General cards */
.panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(23,48,78,.04);
    margin-bottom: 12px;
}

.panel-title {
    color: #183451;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 8px;
}

.panel-text {
    color: #718198;
    font-size: 12px;
    line-height: 1.7;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(100deg, #1769e0, #078e9c);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 800;
    min-height: 43px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    color: white;
    border: 0;
    filter: brightness(1.08);
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 9px;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #425a74;
    font-weight: 700;
    font-size: 13px;
}

/* Prediction results */
.result {
    border-radius: 16px;
    padding: 23px;
    border: 1px solid;
    min-height: 150px;
}

.result-risk {
    background: #fff3f0;
    border-color: #ffd6ce;
    color: #a43d2e;
}

.result-safe {
    background: #edfaf5;
    border-color: #c7ecdd;
    color: #08765c;
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    margin: 10px 0;
}

.result-copy {
    font-size: 12px;
    line-height: 1.7;
}

.note {
    color: #8291a4;
    font-size: 11px;
    line-height: 1.6;
}

.workflow {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.workflow-number {
    color: #1686a5;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
}

.workflow-title {
    color: #193956;
    font-size: 14px;
    font-weight: 800;
    margin: 8px 0;
}

.workflow-text {
    color: #75869a;
    font-size: 11px;
    line-height: 1.6;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    color: #153657;
    font-weight: 800;
}

hr {
    border-color: #e3ebf4;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- LOAD FILES -----------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


missing = [
    filename for filename in [MODEL_PATH, DATA_PATH]
    if not os.path.exists(filename)
]

if missing:
    st.error("Missing required files: " + ", ".join(missing))
    st.info(
        "Make sure app.py, best_churn_model.joblib and "
        "Churn_Modelling_Featured.csv are in the repository root."
    )
    st.stop()

try:
    model = load_model()
    data = load_data()

except Exception as error:
    st.error(f"Could not load project files: {error}")
    st.stop()

if "Exited" not in data.columns:
    st.error("The dataset must contain the Exited target column.")
    st.stop()

X = data.drop(columns=["Exited"])
feature_columns = X.columns.tolist()


# ---------------------- NAVIGATION -----------------------

PAGES = [
    "Home",
    "Predict Churn",
    "Model Performance",
    "About Project"
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:

    st.markdown(
        '<div class="brand-icon">🏦</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-title">BANK CUSTOMER<br>INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">Customer Churn Analytics<br>'
        'Machine Learning Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-label">MAIN WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        PAGES,
        key="page",
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<div class="side-label">PROJECT INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'SELECTED MODEL<br><b>Random Forest</b><br><br>'
        'APPLICATION<br><b>Streamlit Web App</b><br><br>'
        'PROJECT TYPE<br><b>End-to-End Machine Learning</b>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="brand-subtitle">AI & Data Science Graduation Project</div>',
        unsafe_allow_html=True
    )


# ---------------------- COMMON FUNCTIONS -----------------

def show_hero(kicker, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def show_panel(title, description, icon=""):
    st.markdown(
        f"""
        <div class="panel">
            <div style="font-size:25px;margin-bottom:10px">{icon}</div>
            <div class="panel-title">{title}</div>
            <div class="panel-text">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="topbar">
        <span>ANALYTICS PLATFORM / CUSTOMER RETENTION</span>
        <span class="status">● APPLICATION READY</span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: HOME
# =========================================================

if page == "Home":

    show_hero(
        "AI POWERED ANALYTICS",
        "Bank Customer Churn Prediction",
        "Using machine learning to identify customers who may leave "
        "the bank, helping support better customer retention decisions."
    )

    show_section("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("TOTAL CUSTOMERS", f"{len(data):,}", "Prepared dataset"),
        ("MODEL FEATURES", str(len(feature_columns)), "Input variables"),
        ("SELECTED MODEL", "Random Forest", "Best selected classifier"),
        ("TEST ACCURACY", "85.40%", "Final test-set accuracy")
    ]

    for col, (label, value, note) in zip(
        [c1, c2, c3, c4], cards
    ):
        with col:
            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    show_section("Explore the Platform")

    a, b, c = st.columns(3)

    with a:
        show_panel(
            "Customer Prediction",
            "Enter customer information and estimate the probability of churn.",
            "🎯"
        )

    with b:
        show_panel(
            "Model Performance",
            "Explore model evaluation metrics and classification results.",
            "📊"
        )

    with c:
        show_panel(
            "Project Methodology",
            "Learn about the dataset, machine learning workflow and deployment.",
            "⚙️"
        )

    show_section("Project Highlights")

    left, right = st.columns([1.2, 1])

    with left:
        show_panel(
            "From Data to Prediction",
            "This project uses historical bank customer information to train "
            "machine learning models. The selected Random Forest model is "
            "integrated into a web application to provide interactive predictions.",
            "🧠"
        )

    with right:
        show_panel(
            "Technology Stack",
            "Python  •  Pandas  •  Scikit-learn  •  Joblib  •  Streamlit  •  Matplotlib",
            "💻"
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start a Customer Prediction →", use_container_width=True):
            st.session_state.page = "Predict Churn"
            st.rerun()

    with col2:
        if st.button("Explore Model Performance →", use_container_width=True):
            st.session_state.page = "Model Performance"
            st.rerun()


# =========================================================
# PAGE 2: PREDICT CHURN
# =========================================================

elif page == "Predict Churn":

    show_hero(
        "CUSTOMER RISK ASSESSMENT",
        "Predict Customer Churn",
        "Enter customer details to estimate the likelihood "
        "of leaving the bank using the trained Random Forest model."
    )

    show_section("Customer Information")

    st.markdown(
        '<div class="note">Complete the customer profile below. '
        'Available values are based on the prepared project dataset.</div>',
        unsafe_allow_html=True
    )

    values = {}

    with st.form("prediction_form"):

        left, right = st.columns(2)

        for index, column in enumerate(feature_columns):

            series = X[column].dropna()

            if series.empty:
                continue

            label = column.replace("_", " ").title()

            with [left, right][index % 2]:

                if pd.api.types.is_bool_dtype(X[column].dtype):

                    options = [False, True]

                    values[column] = st.selectbox(
                        label,
                        options,
                        format_func=lambda x: "Yes" if x else "No",
                        key=f"input_{column}"
                    )

                elif pd.api.types.is_numeric_dtype(X[column].dtype):

                    unique_values = sorted(series.unique().tolist())

                    if len(unique_values) <= 2:

                        values[column] = st.selectbox(
                            label,
                            unique_values,
                            key=f"input_{column}"
                        )

                    elif pd.api.types.is_integer_dtype(X[column].dtype):

                        values[column] = st.number_input(
                            label,
                            min_value=int(series.min()),
                            max_value=int(series.max()),
                            value=int(series.median()),
                            step=1,
                            key=f"input_{column}"
                        )

                    else:

                        values[column] = st.number_input(
                            label,
                            min_value=float(series.min()),
                            max_value=float(series.max()),
                            value=float(series.median()),
                            format="%.4f",
                            key=f"input_{column}"
                        )

                else:

                    options = series.astype(str).unique().tolist()

                    values[column] = st.selectbox(
                        label,
                        options,
                        key=f"input_{column}"
                    )

        submitted = st.form_submit_button(
            "✦  Generate Churn Prediction",
            use_container_width=True
        )

    if submitted:

        try:

            input_data = pd.DataFrame(
                [[values[column] for column in feature_columns]],
                columns=feature_columns
            )

            for column in feature_columns:

                if pd.api.types.is_numeric_dtype(X[column].dtype):
                    input_data[column] = pd.to_numeric(
                        input_data[column]
                    )

                elif pd.api.types.is_bool_dtype(X[column].dtype):
                    input_data[column] = input_data[column].astype(bool)

            prediction = model.predict(input_data)[0]

            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "The saved model does not support probability prediction."
                )

            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            churn_index = next(
                (
                    i for i, cls in enumerate(classes)
                    if str(cls).strip() in ["1", "1.0", "True"]
                ),
                None
            )

            if churn_index is None:
                raise ValueError("Churn class 1 was not found.")

            churn_probability = float(probabilities[churn_index])

            show_section("Prediction Result")

            result_col, probability_col = st.columns([1.2, 1])

            with result_col:

                if int(prediction) == 1:

                    st.markdown(
                        """
                        <div class="result result-risk">
                            <div class="result-title">
                                ⚠ Customer is Likely to Churn
                            </div>
                            <div class="result-copy">
                                The model predicts that this customer
                                may leave the bank. Further review
                                may be useful for customer retention.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="result result-safe">
                            <div class="result-title">
                                ✓ Customer is Likely to Stay
                            </div>
                            <div class="result-copy">
                                The model predicts that this customer
                                is not likely to leave the bank.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with probability_col:

                st.markdown('<div class="panel">', unsafe_allow_html=True)

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.1%}"
                )

                st.progress(churn_probability)

                st.markdown(
                    '<div class="note">Estimated probability generated '
                    'by the trained Random Forest model.</div></div>',
                    unsafe_allow_html=True
                )

            st.caption(
                "This result is an estimate based on historical data, "
                "not a guarantee of future customer behavior."
            )

        except Exception as error:

            st.error(f"Prediction Error: {error}")

            st.info(
                "Check that the model and dataset use the same feature "
                "columns, order, and preprocessing."
            )


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    show_hero(
        "MODEL VALIDATION & ANALYTICS",
        "Model Performance",
        "Evaluate the selected Random Forest model using "
        "cross-validation and final test-set results."
    )

    show_section("Random Forest - Final Test Results")

    metric_columns = st.columns(5)

    final_metrics = [
        ("ACCURACY", "85.40%"),
        ("PRECISION", "74.89%"),
        ("RECALL", "42.51%"),
        ("F1-SCORE", "54.23%"),
        ("ROC-AUC", "85.24%")
    ]

    for col, (label, value) in zip(metric_columns, final_metrics):

        with col:
            st.metric(label, value)

    st.markdown(
        '<div class="note">The Random Forest model was selected '
        'based on the highest mean F1-score across stratified '
        '5-fold cross-validation.</div>',
        unsafe_allow_html=True
    )

    show_section("Model Evaluation")

    left, right = st.columns(2)

    # Confusion Matrix
    with left:

        st.markdown(
            '<div class="panel"><div class="panel-title">'
            'Confusion Matrix</div><div class="panel-text">'
            'Actual versus predicted customer classes.</div></div>',
            unsafe_allow_html=True
        )

        matrix = [
            [1535, 58],
            [234, 173]
        ]

        fig, ax = plt.subplots(figsize=(6, 4))

        image = ax.imshow(matrix, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stayed", "Churned"])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Stayed", "Churned"])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Random Forest Confusion Matrix")

        for i in range(2):
            for j in range(2):

                ax.text(
                    j, i,
                    str(matrix[i][j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i][j] > 800 else "#183451",
                    fontsize=14,
                    fontweight="bold"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Cross-validation chart
    with right:

        st.markdown(
            '<div class="panel"><div class="panel-title">'
            'Cross-Validation Results</div><div class="panel-text">'
            'Random Forest mean scores across 5 folds.</div></div>',
            unsafe_allow_html=True
        )

        metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ]

        metric_values = [
            0.8521,
            0.7363,
            0.4270,
            0.5404,
            0.8440
        ]

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            metric_names,
            metric_values,
            color=["#1769e0", "#13a6a6", "#7c6be8", "#e8a23b", "#2d9cdb"]
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Random Forest - CV Metrics")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)

        ax.bar_label(bars, fmt="%.2f", padding=3)

        plt.xticks(rotation=20)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    show_section("Model Comparison")

    if os.path.exists(COMPARISON_PATH):

        try:
            comparison = pd.read_csv(COMPARISON_PATH)

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:
            st.warning(
                f"Could not read the comparison file: {error}"
            )

    else:

        st.info(
            "The file model_comparison_results.csv was not found. "
            "Upload it to display the full comparison between "
            "Logistic Regression, KNN, Decision Tree, and Random Forest."
        )

    st.warning(
        "Model limitation: Recall is 42.51%. This means the model "
        "does not identify every customer who actually leaves. "
        "This limitation should be considered when interpreting results."
    )


# =========================================================
# PAGE 4: ABOUT PROJECT
# =========================================================

elif page == "About Project":

    show_hero(
        "PROJECT DOCUMENTATION",
        "About the Project",
        "An end-to-end machine learning project covering "
        "data preparation, exploratory analysis, model development, "
        "evaluation, and web deployment."
    )

    show_section("Project Overview")

    left, right = st.columns(2)

    with left:

        show_panel(
            "Project Objective",
            "Predict whether a bank customer is likely to leave "
            "the bank using historical customer data and machine "
            "learning classification algorithms.",
            "🎯"
        )

    with right:

        show_panel(
            "Target Variable",
            "Exited is the target column. A value of 0 represents "
            "a customer who stayed, while 1 represents a customer "
            "who left the bank.",
            "🏷️"
        )

    show_section("Project Workflow")

    workflow = [
        (
            "01",
            "Data Collection",
            "Obtain the bank customer dataset."
        ),
        (
            "02",
            "Data Preprocessing",
            "Handle missing values, duplicates and data quality."
        ),
        (
            "03",
            "Exploratory Data Analysis",
            "Explore customer characteristics and churn patterns."
        ),
        (
            "04",
            "Model Development",
            "Train and compare four classification algorithms."
        ),
        (
            "05",
            "Model Evaluation",
            "Evaluate the models using cross-validation and test data."
        ),
        (
            "06",
            "Deployment",
            "Integrate the selected model into a Streamlit application."
        )
    ]

    workflow_columns = st.columns(3)

    for index, (number, title, description) in enumerate(workflow):

        with workflow_columns[index % 3]:

            st.markdown(
                f"""
                <div class="workflow">
                    <div class="workflow-number">STAGE {number}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    show_section("Dataset Information")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric("Total Records", f"{len(data):,}")

    with d2:
        st.metric("Input Features", len(feature_columns))

    with d3:
        st.metric("Target Variable", "Exited")

    with st.expander("View Dataset Feature Names"):
        st.write(feature_columns)

    show_section("Tools & Technologies")

    t1, t2, t3 = st.columns(3)

    with t1:
        show_panel(
            "Programming",
            "Python",
            "🐍"
        )

    with t2:
        show_panel(
            "Machine Learning",
            "Pandas, Scikit-learn, Joblib",
            "🧠"
        )

    with t3:
        show_panel(
            "Deployment",
            "Streamlit, GitHub, Streamlit Cloud",
            "☁️"
        )

    show_section("Responsible Use")

    show_panel(
        "Important Note",
        "The model provides estimated predictions based on historical "
        "data. Results should support further analysis and human "
        "decision-making rather than automatically determine how "
        "customers are treated.",
        "ℹ️"
    )


# ---------------------- FOOTER ---------------------------

st.markdown("<br><hr>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8190a3;
        font-size:11px;
        padding:12px;
    ">
        BANK CUSTOMER CHURN PREDICTION
        &nbsp; • &nbsp;
        MACHINE LEARNING PROJECT
        &nbsp; • &nbsp;
        STREAMLIT
    </div>
    """,
    unsafe_allow_html=True
)
