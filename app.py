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
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f4f7fb; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071d3b 0%, #0b2b52 100%);
    }
    [data-testid="stSidebar"] * { color: #f3f7ff !important; }

    .brand { font-size: 1.25rem; font-weight: 800; line-height: 1.25; color: white; }
    .brand-sub { color: #b7c9e3 !important; font-size: .82rem; margin-top: .45rem; }

    .hero {
        padding: 2.1rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #061a33 0%, #0d3b66 50%, #104f7a 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px rgba(9,38,75,.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero h1 { color: white !important; font-size: 2.25rem; margin: .25rem 0 .6rem 0; }
    .hero p { color: #e1edf9 !important; font-size: 1rem; max-width: 800px; }

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

    .field-label {
        color: #102d52;
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        margin-top: 0.6rem;
    }

    .soft-card {
        background: #ffffff;
        border: 1px solid #e2eaf4;
        border-radius: 17px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 5px 18px rgba(25,55,90,.045);
        color: #102d52;
    }

    .soft-card h3 { color: #102d52 !important; }
    .soft-card p, .soft-card li { color: #334e68 !important; }
    .muted { color: #486581 !important; }

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
