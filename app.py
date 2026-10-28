
import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# BANK CUSTOMER CHURN PREDICTION - WEB APPLICATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "Churn_Modelling_Featured.csv"

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer information to predict whether "
    "the customer is likely to leave the bank."
)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    model = load_model(MODEL_PATH)
    data = load_data(DATA_PATH)
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

if "Exited" not in data.columns:
    st.error("The dataset must contain the 'Exited' column.")
    st.stop()

# Use the same input columns used during model training.
feature_data = data.drop(columns=["Exited"])
feature_columns = feature_data.columns.tolist()

st.caption("Model: trained Random Forest / selected model pipeline")

# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------

st.subheader("Customer Information")

customer_values = {}

with st.form("customer_form"):

    for column in feature_columns:

        series = feature_data[column].dropna()

        if series.empty:
            st.error(f"No available values for {column}.")
            st.stop()

        # Categorical features
        if (
            pd.api.types.is_object_dtype(feature_data[column])
            or isinstance(feature_data[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(feature_data[column])
        ):
            options = series.unique().tolist()

            customer_values[column] = st.selectbox(
                column,
                options=options
            )

        # Numeric features
        elif pd.api.types.is_numeric_dtype(feature_data[column]):

            unique_values = sorted(series.unique().tolist())

            # Binary features such as HasCrCard and HasBalance
            if len(unique_values) <= 2:
                customer_values[column] = st.selectbox(
                    column,
                    options=unique_values
                )

            # Integer features
            elif pd.api.types.is_integer_dtype(feature_data[column]):
                default_value = int(series.median())

                customer_values[column] = st.number_input(
                    column,
                    min_value=int(series.min()),
                    max_value=int(series.max()),
                    value=default_value,
                    step=1
                )

            # Continuous numeric features
            else:
                customer_values[column] = st.number_input(
                    column,
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                    value=float(series.median())
                )

        
        else:
            # Treat remaining columns as categorical inputs
            options = series.astype(str).unique().tolist()

            customer_values[column] = st.selectbox(
                column,
                options=options
            )


    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    input_df = pd.DataFrame(
        [customer_values],
        columns=feature_columns
    )

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        # Find the probability belonging to class 1.
        classes = list(model.classes_)
        churn_index = classes.index(1)
        churn_probability = float(probabilities[churn_index])

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "This customer is predicted to leave the bank."
            )
        else:
            st.success(
                "This customer is predicted to stay with the bank."
            )

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1%}"
        )

        st.progress(churn_probability)

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
