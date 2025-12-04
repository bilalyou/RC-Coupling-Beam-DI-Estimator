# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 19:03:21 2025

@author: youni
"""

# ======================================================================
# IMPORTS
# ======================================================================
import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb
import numpy as np
from tensorflow.keras.models import load_model
import base64
from pathlib import Path
import matplotlib.pyplot as plt   # <<< ADDED for plotting

# ======================================================================
# PAGE CONFIG + CSS
# ======================================================================
st.set_page_config(page_title="PADI Prediction", layout="wide", page_icon="")

# --- Custom CSS for Styling ---
st.markdown(r"""
<style>
    .block-container { padding-top: 2rem; }
    .stNumberInput > div > div, .stSelectbox > div > div {
        max-width: 240px !important;
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 40px !important;
        font-weight: 1000;
    }
    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .form-banner {         
        text-align: center;
        background: linear-gradient(lightgray);
        padding: 0.4rem;
        font-size: 30px;
        font-weight: 800;
        color: black;
        border-radius: 0px;
        margin: 0rem 0;
    }
    .prediction-result { 
        font-size: 25px;
        font-weight: bold;
        color: black;
        background-color: lightgray;
        padding: 0.2rem;
        border-radius: 0px;
        text-align: left;
        margin-top: 0rem;
    }
    .recent-box {
        background-color: #f8f9fa;
        padding: 0.6rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        border-left: 4px solid #4CAF50;
        font-weight: 600;
    }
    div.stButton > button {
        background-color: #2ecc71;
        color: black;
        font-weight: bold;
        font-size: 20px;
        border-radius: 0px;
        padding: 0rem 3.0rem;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #27ae60;
    }
    div.stButton:nth-of-type(3) > button {
        background-color: #f28b82 !important;
        color: white !important;
        font-weight: bold !important;
    }
    div.stButton:nth-of-type(3) > button:hover {
        background-color: #e06666 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# MODELS & HELPERS
# ======================================================================
ann_ps_model = load_model("ANN_PS_Model.keras")
ann_ps_scaler_X = joblib.load("ANN_PS_Scaler_X.save")
ann_ps_scaler_y = joblib.load("ANN_PS_Scaler_y.save")

ann_mlp_model = load_model("ANN_MLP_Model.keras")
ann_mlp_scaler_X = joblib.load("ANN_MLP_Scaler_X.save")
ann_mlp_scaler_y = joblib.load("ANN_MLP_Scaler_y.save")

et_model = joblib.load("Best_ExtraTrees_Model.joblib")


def normalize_input(x_raw, scaler):
    return scaler.transform(x_raw)


def denormalize_output(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1))[0][0]


@st.cache_resource
def load_models():
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("Best_XGBoost_Model.json")

    cat_model = catboost.CatBoostRegressor()
    cat_model.load_model("Best_CatBoost_Model.cbm")

    lgb_model = lgb.Booster(model_file="Best_LightGBM_Model.txt")

    return {
        "CatBoost": cat_model,           # first in dropdown
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "Extra Trees": et_model,
        "PS": ann_ps_model,
        "MLP": ann_mlp_model
    }


models = load_models()

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# ======================================================================
# MAIN TWO-COLUMN LAYOUT
# ======================================================================
col1, col2 = st.columns([2.2, 1.5], gap="large")

# ----------------------------- COLUMN 1 -------------------------------
with col1:
    logo_path = Path("logo2-01.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            base64_logo = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 10px;'>
                <img src='data:image/png;base64,{base64_logo}' width='550'>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<h1 style='text-align: center;'>Damage Estimation Interface</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align: center;'>"
        "This online app estimates the Park-Ang damage index of RC coupling beams."
        "by providing only the relevant input parameters."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='form-banner'>Input Parameters</div>",
        unsafe_allow_html=True
    )
    st.session_state.input_error = False

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        L = st.number_input("Beam Length $l$ (mm)", value=1000.0,
                            min_value=424.0, max_value=2235.0, step=1.0)
        h = st.number_input("Beam Height $h$ (mm)", value=400.0,
                            min_value=169.0, max_value=880.0, step=1.0)
        b = st.number_input("Beam Width $b$ (mm)", value=200.0,
                            min_value=100.0, max_value=406.0, step=1.0)
        AR = st.number_input("Aspect Ratio $l/h$", value=2.5,
                             min_value=0.75, max_value=4.9, step=0.01)
        fc = st.number_input("Concrete Strength $f'_c$ (MPa)", value=54.0,
                             min_value=18.1, max_value=86.0, step=0.1)

    with c2:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        fyl = st.number_input("Yield Strength of Longitudinal Bars $f_{yl}$ (MPa)",
                              value=476.0, min_value=281.0, max_value=827.0, step=1.0)
        fyv = st.number_input("Yield Strength of Stirrups $f_{yv}$ (MPa)",
                              value=331.0, min_value=212.0, max_value=953.0, step=1.0)
        fyd = st.number_input("Yield Strength of Diagonal Bars $f_{yd}$ (MPa)",
                              value=476.0, min_value=0.0, max_value=883.0, step=1.0)
        Pl = st.number_input("Longitudinal Reinforcement $\\rho_l$ (%)",
                             value=0.25, min_value=0.09, max_value=4.1, step=0.01)
        Pv = st.number_input("Stirrups Reinforcement $\\rho_v$ (%)",
                             value=0.21, min_value=0.096, max_value=2.9, step=0.001)

    with c3:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        s = st.number_input("Stirrup Spacing $s$ (mm)", value=150.0,
                            min_value=25.0, max_value=500.0, step=1.0)
        Pd = st.number_input("Diagonal Reinforcement $\\rho_d$ (%)",
                             value=1.005, min_value=0.0, max_value=5.8, step=0.01)
        alpha = st.number_input("Diagonal Angle $\\alpha$", value=17.5,
                                min_value=0.0, max_value=45.0, step=1.0)
        drift = st.number_input("$\\theta$ (%)", value=1.5,
                                min_value=0.06, max_value=12.22, step=0.1)

# ----------------------------- COLUMN 2 -------------------------------
with col2:
    st.image("beam-01.svg", width=600)
    st.markdown(
        "<div style='text-align:center; font-weight:800; font-size:18px;'>"
        "RC Coupling Beam Configurations</div>",
        unsafe_allow_html=True
    )

    model_choice = st.selectbox("Model Selection", list(models.keys()))

    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.2, 1.2])
    with c_btn1:
        submit = st.button("Predict")
    with c_btn2:
        if st.button("Reset"):
            st.rerun()
    with c_btn3:
        if "results_df" in st.session_state and not st.session_state.results_df.empty:
            csv_data = st.session_state.results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Save Prediction",
                data=csv_data,
                file_name="PADI_predictions.csv",
                mime="text/csv",
                key="download_button"
            )

    # ==================== PREDICTION LOGIC (with plot) ====================
    if submit and not st.session_state.input_error:
        # include drift as last feature
        input_array = np.array(
            [[L, h, b, AR, fc, fyl, fyv, Pl, Pv, s, Pd, fyd, alpha, drift]]
        )

        input_df = pd.DataFrame(
            input_array,
            columns=['L', 'h', 'b', 'AR', 'f′c', 'fyl', 'fyv',
                     'Pl', 'Pv', 's', 'Pd', 'fyd', 'α֯', 'θ']
        )

        model = models[model_choice]

        if model_choice == "LightGBM":
            pred = model.predict(input_df)[0]
        elif model_choice == "PS":
            input_norm = normalize_input(input_array, ann_ps_scaler_X)
            pred_scaled = model.predict(input_norm)[0][0]
            pred = denormalize_output(pred_scaled, ann_ps_scaler_y)
        elif model_choice == "MLP":
            input_norm = normalize_input(input_array, ann_mlp_scaler_X)
            pred_scaled = model.predict(input_norm)[0][0]
            pred = denormalize_output(pred_scaled, ann_mlp_scaler_y)
        elif model_choice == "Extra Trees":
           # <<< use raw numpy array to avoid feature-name mismatch
           pred = model.predict(input_array)[0]
        else:
            # CatBoost, XGBoost, Extra Trees
            pred = model.predict(input_df)[0]

        input_df["Predicted_PADI"] = pred
        st.session_state.results_df = pd.concat(
            [st.session_state.results_df, input_df],
            ignore_index=True
        )

        # numeric prediction text
        st.markdown(
            f"<div class='prediction-result'>"
            f"Predicted Damage Index : {pred:.2f} </div>",
            unsafe_allow_html=True
        )

        # -------- Damage state classification --------
        if pred < 0.2:
            damage_state = "No damage"
            state_color = "green"
        elif pred < 0.5:
            damage_state = "Partial damage"
            state_color = "orange"
        elif pred <= 1.0:
            damage_state = "Severe damage"
            state_color = "red"
        else:
            damage_state = "Collapse"
            state_color = "gray"

        # -------- Plot: Drift vs Damage Index --------
        fig, ax = plt.subplots(figsize=(3, 2))   # compact plot to fit space

        # fixed axis ranges
        x_max = 12.20          # covers full drift range (0–12.22%)
        y_max = 1.5            # DI range

        # diagonal line from origin to predicted point
        ax.plot([0, drift], [0, pred], linewidth=2, color="black")

        # ---- marker plotted AFTER line to keep on top ----
        ax.scatter([drift], [pred],
                   s=40,
                   facecolors="white",
                   edgecolors=state_color,
                   linewidths=1.5,
                   zorder=5)

        # ---- predicted value label to the RIGHT of circle ----
        ax.text(drift + 0.5,                 # small horizontal offset
                pred,                         # same vertical level as circle
                f"{pred:.2f}",
                ha="left", va="center",
                fontsize=8,
                color=state_color,
                fontweight="bold",
                zorder=6)


        # axis settings
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("$\\theta$ (%)", fontsize=10)
        ax.set_ylabel("Damage Index", fontsize=10)
        ax.set_yticks(np.linspace(0.0, y_max, 6))

        # remove top and right frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ticks
        ax.tick_params(labelsize=5)

        # light grid
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        # damage state text inside plot (top-center)
        ax.text(x_max * 0.5, y_max * 0.92,
                damage_state,
                fontsize=14, fontweight="bold",
                color=state_color,
                ha="center", va="center")

        plt.tight_layout()
        st.pyplot(fig)





# ======================================================================
# FOOTER
# ======================================================================
st.markdown("""
<hr style='margin-top: 3rem;'>
<div style='text-align: center; color: gray; font-size: 18px;'>
    Developed by [Bilal Younis]. For academic and research purposes only.
</div>
""", unsafe_allow_html=True)


