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

# Visual styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f4f7fb; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071d3b 0%, #0b2b52 100%);
    }
    [data-testid="stSidebar"] * { color: #f3f7ff; }
    .brand { font-size: 1.25rem; font-weight: 800; line-height: 1.25; }
    .brand-sub { color: #b7c9e3 !important; font-size: .82rem; margin-top: .45rem; }
    .hero {
        padding: 2.1rem 2.2rem; border-radius: 22px;
        background: linear-gradient(115deg, #09264b 0%, #12558a 65%, #167b91 100%);
        color: white; margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(9,38,75,.15);
    }
    .hero h1 { color: white; font-size: 2.25rem; margin: .25rem 0 .6rem 0; }
    .hero p { color: #e1edf9; font-size: 1rem; max-width: 800px; }
    .eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .75rem; font-weight: 700; color: #9edee5; }
    .section-title { color: #102d52; font-size: 1.35rem; font-weight: 800; margin: .5rem 0 1rem 0; }
    .soft-card {
        background: white; border: 1px solid #e2eaf4; border-radius: 17px;
        padding: 1.15rem 1.25rem; box-shadow: 0 5px 18px rgba(25,55,90,.045);
    }
    .metric-label { color: #61748d; font-size: .84rem; font-weight: 600; }
    .metric-value { color: #102d52; font-size: 1.75rem; font-weight: 800; margin-top: .2rem; }
    .muted { color: #61748d; }
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
