
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "Churn_Modelling_Featured.csv"
RESULTS_PATH = "model_comparison_results.csv"

# ------------------------------------------------------------
# CUSTOM DESIGN & CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071d3b 0%, #0b2b52 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f3f7ff !important;
    }

    .brand {
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.25;
        color: white;
    }

    .brand-sub {
        color: #b7c9e3 !important;
        font-size: .82rem;
        margin-top: .45rem;
    }

    .hero {
        padding: 2.1rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #061a33 0%, #0d3b66 50%, #104f7a 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(9,38,75,.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero h1 {
        color: white !important;
        font-size: 2.25rem;
        margin: .25rem 0 .6rem 0;
    }

    .hero p {
        color: #e1edf9 !important;
        font-size: 1rem;
        max-width: 800px;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .15em;
        font-size: .75rem;
        font-weight: 700;
        color: #9edee5;
    }

    .section-title {
        color: #102d52;
        font-size: 1.35rem;
        font-weight: 800;
        margin: .5rem 0 1rem 0;
    }

    .soft-card {
        background: #ffffff;
        border: 1px solid #e2eaf4;
        border-radius: 17px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 5px 18px rgba(25,55,90,.045);
        color: #102d52;
    }

    .soft-card h3 {
        color: #102d52 !important;
    }

    .soft-card p, .soft-card li {
        color: #334e68 !important;
    }

    .muted {
        color: #486581 !important;
    }

    .result-good {
        border-radius: 18px;
        padding: 1.5rem;
        background: #e9faf4;
        border: 1px solid #b7ead8;
        color: #087b5b;
    }

    .result-risk {
        border-radius: 18px;
        padding: 1.5rem;
        background: #fff1ee;
        border: 1px solid #ffd1c8;
        color: #b53d2e;
    }

    .result-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin: .35rem 0;
    }

    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #1769e0, #1389a0);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
    }

    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        color: white;
        border: 0;
        filter: brightness(1.05);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2eaf4;
        border-radius: 15px;
        padding: 1rem 0.8rem;
    }

    [data-testid="stMetric"] label {
        color: #61748d !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #102d52 !important;
        font-size: 1.45rem !important;
    }

    h2, h3 {
        color: #102d52;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD MODEL AND DATA
# ------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    data = load_data(DATA_PATH)

except Exception as error:
    st.error(f"Could not load model or dataset: {error}")
    st.stop()

if "Exited" not in data.columns:
    st.error("The dataset must contain the target column Exited.")
    st.stop()

feature_data = data.drop(columns=["Exited"])
feature_columns = feature_data.columns.tolist()


# ------------------------------------------------------------
# FRIENDLY FEATURE NAMES AND DESCRIPTIONS
# ------------------------------------------------------------

feature_labels = {
    "CreditScore": "Credit Score",
    "Geography": "Country",
    "Gender": "Gender",
    "Age": "Customer Age",
    "Tenure": "Years with Bank",
    "Balance": "Account Balance",
    "NumOfProducts": "Number of Bank Products",
    "HasCrCard": "Credit Card Ownership",
    "IsActiveMember": "Active Membership",
    "EstimatedSalary": "Estimated Annual Salary",
    "BalanceSalaryRatio": "Balance to Salary Ratio",
    "AgeGroup": "Customer Age Group",
    "CreditScoreGroup": "Credit Score Group",
    "TenureGroup": "Customer Tenure Group",
}

feature_descriptions = {
    "CreditScore": "التقييم الائتماني للعميل. | The customer's credit score.",
    "Geography": "الدولة التي يقيم فيها العميل. | The customer's country.",
    "Gender": "جنس العميل. | The customer's gender.",
    "Age": "عمر العميل بالسنوات. | The customer's age in years.",
    "Tenure": "عدد سنوات تعامل العميل مع البنك. | Years the customer has been with the bank.",
    "Balance": "رصيد حساب العميل في البنك. | The customer's bank account balance.",
    "NumOfProducts": "عدد منتجات البنك التي يستخدمها العميل. | Number of bank products used.",
    "HasCrCard": "هل يمتلك العميل بطاقة ائتمانية؟ | Does the customer have a credit card?",
    "IsActiveMember": "هل العميل عضو نشط في البنك؟ | Is the customer an active bank member?",
    "EstimatedSalary": "الراتب السنوي التقديري للعميل. | The customer's estimated annual salary.",
    "BalanceSalaryRatio": "نسبة رصيد الحساب إلى الراتب. | Account balance compared with salary.",
    "AgeGroup": "الفئة العمرية للعميل. | The customer's age category.",
    "CreditScoreGroup": "فئة التقييم الائتماني للعميل. | The customer's credit score category.",
    "TenureGroup": "فئة سنوات تعامل العميل مع البنك. | The customer's tenure category.",
}


# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
pages = [
    "Home",
    "Predict Churn",
    "Model Performance",
    "About Project"
]


def go_to_predict():
    st.session_state["nav_radio"] = "Predict Churn"


if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "Home"


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:

    st.markdown("## 🏦 Bank Churn")

    st.markdown(
        '<div class="brand">PREDICTOR</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-sub">Customer retention through data-driven insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        pages,
        key="nav_radio",
        label_visibility="visible",
    )

    st.markdown("---")

    st.caption("Machine Learning • Random Forest")
    st.caption("A student project for customer churn analysis")


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Customer intelligence platform</div>
        <h1>Bank Customer Churn Prediction</h1>
        <p>
        Understand customer churn risk with a machine-learning model.
        Explore a customer profile, estimate churn probability, and use
        the insights to support proactive retention decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project at a glance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customer records", f"{len(data):,}")
    c2.metric("Input features", f"{len(feature_columns)}")
    c3.metric("Selected model", "Random Forest")
    c4.metric("Test accuracy", "85.40%")

    left, right = st.columns([1.15, 1])

    with left:

        st.markdown("""
        <div class="soft-card">
            <h3>🎯 What this application does</h3>
            <p class="muted">
            Enter customer information and receive a model-based prediction
            of whether the customer may leave the bank.
            </p>
            <ul>
              <li>Creates a prediction from the saved trained model.</li>
              <li>Displays the estimated probability of churn.</li>
              <li>Summarizes the model's evaluation metrics.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.button(
            "Start a customer prediction →",
            use_container_width=True,
            on_click=go_to_predict
        )

    with right:

        st.markdown("""
        <div class="soft-card">
            <h3>📌 Why churn prediction matters</h3>
            <p class="muted">
            Finding customers who may be at risk can help a bank prioritize
            follow-up and design customer-retention strategies.
            </p>
            <hr>
            <p><b>Important:</b> Predictions are estimates from a
            machine-learning model, not guarantees of future behavior.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Use Predict Churn in the sidebar to enter customer details, "
        "or Model Performance to review evaluation results."
    )


# ------------------------------------------------------------
# PREDICT CHURN PAGE
# ------------------------------------------------------------
elif page == "Predict Churn":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Customer risk assessment</div>
        <h1>Predict Customer Churn</h1>
        <p>
        Fill in the customer details below.
        Each field includes a clear name and description.
        The saved model will estimate the likelihood of churn.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Customer Information</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Please enter the customer's information. "
        "Each field has an English name and an Arabic explanation."
    )

    customer_values = {}

    with st.form("customer_form", clear_on_submit=False):

        cols = st.columns(2)

        for index, column in enumerate(feature_columns):

            series = feature_data[column].dropna()

            if series.empty:
                st.error(f"No available values for {column}.")
                st.stop()

            # Friendly name for the input
            label = feature_labels.get(
                column,
                column.replace("_", " ").replace("-", " ").title()
            )

            # Explanation for the input
            description = feature_descriptions.get(
                column,
                f"أدخل قيمة {label}. | Enter the value for {label.lower()}."
            )

            with cols[index % 2]:

                st.markdown(f"**{label}**")
                st.caption(description)

                is_categorical = (
                    pd.api.types.is_object_dtype(feature_data[column])
                    or isinstance(
                        feature_data[column].dtype,
                        pd.CategoricalDtype
                    )
                    or pd.api.types.is_bool_dtype(feature_data[column])
                )

                if is_categorical:

                    options = series.astype(str).unique().tolist()

                    customer_values[column] = st.selectbox(
                        label,
                        options=options,
                        key=f"input_{column}",
                        label_visibility="collapsed",
                    )

                elif pd.api.types.is_numeric_dtype(feature_data[column]):

                    unique_values = sorted(series.unique().tolist())

                    if len(unique_values) <= 2:

                        customer_values[column] = st.selectbox(
                            label,
                            options=unique_values,
                            format_func=lambda x: (
                                str(int(x))
                                if isinstance(x, (int, float))
                                and float(x).is_integer()
                                else str(x)
                            ),
                            key=f"input_{column}",
                            label_visibility="collapsed",
                        )

                    elif pd.api.types.is_integer_dtype(feature_data[column]):

                        customer_values[column] = st.number_input(
                            label,
                            min_value=int(series.min()),
                            max_value=int(series.max()),
                            value=int(series.median()),
                            step=1,
                            key=f"input_{column}",
                            label_visibility="collapsed",
                        )

                    else:

                        customer_values[column] = st.number_input(
                            label,
                            min_value=float(series.min()),
                            max_value=float(series.max()),
                            value=float(series.median()),
                            key=f"input_{column}",
                            label_visibility="collapsed",
                        )

                else:

                    options = series.astype(str).unique().tolist()

                    customer_values[column] = st.selectbox(
                        label,
                        options=options,
                        key=f"input_{column}",
                        label_visibility="collapsed",
                    )

        submitted = st.form_submit_button(
            "✨ Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [customer_values],
            columns=feature_columns
        )

        for column in feature_columns:

            if pd.api.types.is_numeric_dtype(feature_data[column]):

                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce"
                )

        try:

            prediction = int(model.predict(input_df)[0])

            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            churn_probability = float(
                probabilities[classes.index(1)]
            )

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            result_col, probability_col = st.columns([1.1, 1])

            with result_col:

                if prediction == 1:

                    st.markdown("""
                    <div class="result-risk">
                      <div>⚠️ HIGHER CHURN CONCERN</div>
                      <div class="result-title">Predicted to leave</div>
                      <div>
                      The model classifies this customer as likely to churn.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-good">
                      <div>✓ RETENTION OUTLOOK</div>
                      <div class="result-title">Predicted to stay</div>
                      <div>
                      The model classifies this customer as likely to remain
                      with the bank.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            with probability_col:

                st.markdown(
                    '<div class="soft-card">',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Estimated churn probability",
                    f"{churn_probability:.1%}"
                )

                st.progress(churn_probability)

                st.caption(
                    "Probability estimated by the trained model."
                )

                st.markdown("</div>", unsafe_allow_html=True)

            st.caption(
                "This is a model prediction, not a guarantee "
                "of the customer's future behavior."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")

            st.info(
                "Check that the model and dataset were created "
                "using the same feature columns."
            )


# ------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ------------------------------------------------------------
elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Evaluation and validation</div>
        <h1>Model Performance</h1>
        <p>
        Results from the final test set and cross-validation comparison
        of the trained classifiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Random Forest — test set</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Accuracy", "85.40%")
    m2.metric("Precision", "74.89%")
    m3.metric("Recall", "42.51%")
    m4.metric("F1-score", "54.23%")
    m5.metric("ROC-AUC", "85.24%")

    st.caption(
        "The model was selected using the highest mean F1-score "
        "across stratified 5-fold cross-validation."
    )

    left, right = st.columns(2)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------
    with left:

        st.markdown(
            '<div class="soft-card"><h3>Confusion matrix</h3>'
            '<p class="muted">Actual vs. predicted labels on the 2,000-row test set.</p></div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(5.2, 3.8))

        matrix = [[1535, 58], [234, 173]]

        image = ax.imshow(
            matrix,
            interpolation="nearest",
            cmap="Blues"
        )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")

        ax.set_xticks([0, 1], labels=["Stayed (0)", "Churned (1)"])
        ax.set_yticks([0, 1], labels=["Stayed (0)", "Churned (1)"])

        threshold = max(max(row) for row in matrix) / 2

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(matrix[i][j]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white" if matrix[i][j] > threshold else "#102d52"
                )

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    # --------------------------------------------------------
    # MODEL COMPARISON FROM CSV
    # --------------------------------------------------------
    with right:

        st.markdown(
            '<div class="soft-card"><h3>Cross-validation F1-score</h3>'
            '<p class="muted">Mean F1-score for each candidate model.</p></div>',
            unsafe_allow_html=True
        )

        if os.path.exists(RESULTS_PATH):

            try:

                results = pd.read_csv(RESULTS_PATH)

                required_columns = ["Model", "CV F1"]

                if all(column in results.columns for column in required_columns):

                    names = results["Model"]
                    scores = results["CV F1"]

                    fig, ax = plt.subplots(figsize=(5.2, 3.8))

                    bars = ax.bar(
                        names,
                        scores,
                        color="#1769e0"
                    )

                    ax.set_ylim(0, 0.65)
                    ax.set_ylabel("Mean F1-score")
                    ax.set_title("Model Comparison")
                    ax.tick_params(axis="x", rotation=18)

                    ax.bar_label(
                        bars,
                        fmt="%.3f",
                        padding=3
                    )

                    fig.tight_layout()

                    st.pyplot(fig, use_container_width=True)

                    plt.close(fig)

                else:

                    st.warning(
                        "The comparison CSV must contain Model and CV F1 columns."
                    )

            except Exception as error:

                st.error(
                    f"Could not read model comparison file: {error}"
                )

        else:

            st.warning(
                "Model comparison file not found. "
                "Please add model_comparison_results.csv to the project folder."
            )

    st.markdown(
        '<div class="section-title">Model Limitations & Challenges (Demo Day Review)</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="soft-card">
        <h3>⚠️ Challenges and Future Improvements</h3>
        <ul>
            <li><b>Class Imbalance:</b> Churned customers represent a smaller portion of the dataset, which affects the model's Recall (42.51%).</li>
            <li><b>False Negatives:</b> The model occasionally misses customers who actually churn, highlighting an area for future threshold tuning or cost-sensitive learning.</li>
            <li><b>Actionable Insight:</b> Combining these predictions with customer service follow-ups can help mitigate potential churn risks effectively.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# ABOUT PROJECT PAGE
# ------------------------------------------------------------
elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Project documentation</div>
        <h1>About the Project</h1>
        <p>
        An end-to-end machine-learning project focused on identifying
        bank customers who may leave.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project workflow</div>',
        unsafe_allow_html=True
    )

    steps = st.columns(4)

    workflow = [
        ("01", "Data cleaning",
         "Handled missing values, duplicates, and data-quality issues."),
        ("02", "EDA & features",
         "Explored churn patterns and created useful features."),
        ("03", "Modeling",
         "Compared four classifiers using stratified cross-validation."),
        ("04", "Deployment",
         "Built an interactive Streamlit application for predictions."),
    ]

    for col, (number, title, description) in zip(steps, workflow):

        with col:

            st.markdown(
                f'<div class="soft-card">'
                f'<div class="eyebrow">{number}</div>'
                f'<h3>{title}</h3>'
                f'<p class="muted">{description}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("")

    st.markdown("""
    <div class="soft-card">
      <h3>🎯 Project objective</h3>
      <p class="muted">
      Predict whether a bank customer is likely to leave, helping inform
      customer-retention decisions.
      </p>

      <h3>🧰 Technologies</h3>
      <p class="muted">
      Python · Pandas · Scikit-learn · Joblib · Streamlit · Matplotlib
      </p>

      <h3>⚠️ Responsible use</h3>
      <p class="muted">
      The output is a statistical estimate and should be considered
      alongside human review and other relevant information.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")

st.markdown(
    '<div style="text-align:center;color:#71839a;'
    'font-size:.8rem;padding:.4rem 0 1rem 0;">'
    'Bank Customer Churn Prediction · Machine Learning Project · '
    'Built with Streamlit</div>',
    unsafe_allow_html=True
)
