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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io
# ======================================================================
# PAGE CONFIG + CSS
# ======================================================================
st.set_page_config(page_title="PADI Prediction", layout="wide", page_icon="")

st.markdown(r"""
<style>
    .block-container { padding-top: 3rem; }

    .stNumberInput > div > div, .stSelectbox > div > div {
        max-width: 240px !important;
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 50px !important;
        font-weight: 1000;
    }
    
    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .form-banner {
        text-align: center;
        background: linear-gradient(powderblue);
        padding: 0.4rem;
        font-size: 50px;
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

    /* Buttons (keep your look) */
    div.stButton > button {
        background-color: powderblue;
        color: black;
        font-weight: bold;
        font-size: 20px;
        border-radius: 5px;
        padding: 0rem 3.0rem;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #27ae60;
    }

    /* ===== MAIN 2-COLUMN ROW ONLY (col1 background) ===== */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2))
  > div[data-testid="column"]:first-child
  > div {
    background-color: powderblue;
    border-radius: 16px;
    padding: 20px 20px;
    border: 1px solid white;
}

/* ===== CANCEL the background for any nested columns (buttons etc.) ===== */
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] 
  > div[data-testid="column"] > div {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}



    /* (Optional) If you want right side untouched, do nothing.
       If you ever want same style on right too, create .right-panel similarly. */
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
col1, col2 = st.columns([2, 1.2], gap="large")

# ----------------------------- COLUMN 1 -------------------------------
with col1:
    # ✅ Only LEFT side gets background (no more global column styling)
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)

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
        "This online app estimate the Park-Ang damage index of RC coupling beams "
        "by providing only the relevant input parameters."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='form-banner'>Input Parameters</div>", unsafe_allow_html=True)
    st.session_state.input_error = False

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        L = st.number_input("Beam Length $l$ (mm)", value=1000.0,
                            min_value=400.0, max_value=2200.0, step=1.0)
        h = st.number_input("Beam Height $h$ (mm)", value=400.0,
                            min_value=200.0, max_value=800.0, step=1.0)
        b = st.number_input("Beam Width $b$ (mm)", value=200.0,
                            min_value=150.0, max_value=400.0, step=1.0)
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
        drift = st.number_input("Chord Rotation $\\theta$ (%)", value=1.5,
                                min_value=0.06, max_value=12.22, step=0.1)

    st.markdown("</div>", unsafe_allow_html=True)  # ✅ close left-panel


# ----------------------------- COLUMN 2 -------------------------------
with col2:
    # Right side remains independent (no forced background now)

    # Read the SVG as bytes and convert to base64
    with open("beam-01.svg", "rb") as f:
        svg_bytes = f.read()
    svg_base64 = base64.b64encode(svg_bytes).decode("utf-8")

    img_html = f"""
    <div style='text-align:center;'>
        <img src="data:image/svg+xml;base64,{svg_base64}" width="700">
    </div>
    """
    st.markdown(img_html, unsafe_allow_html=True)

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
            pred = model.predict(input_array)[0]
        else:
            pred = model.predict(input_df)[0]

        input_df["Predicted_PADI"] = pred
        st.session_state.results_df = pd.concat(
            [st.session_state.results_df, input_df],
            ignore_index=True
        )

        # ==================================================================
        # PLOT (your same plot logic)
        # ==================================================================
        fig, ax = plt.subplots(figsize=(2.3, 1.6))

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        y_max = 1.5
        drift_safe = max(float(drift), 1e-6)
        pad = 0.1 * drift_safe
        x_max = drift_safe + pad

        ax.axhspan(0.0, 0.2, facecolor="green",  alpha=0.4, zorder=0)
        ax.axhspan(0.2, 0.5, facecolor="orange", alpha=0.4, zorder=0)
        ax.axhspan(0.5, 1.0, facecolor="red",    alpha=0.4, zorder=0)
        ax.axhspan(1.0, y_max, facecolor="gray", alpha=0.8, zorder=0)

        theta_vals = np.linspace(0.0, drift_safe, 80)
        di_vals = (pred / drift_safe) * theta_vals
        ax.plot(theta_vals, di_vals, linewidth=1, color="black", zorder=3)

        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, y_max)
        ax.set_xlabel("$\\theta$ (%)", fontsize=8)
        ax.set_ylabel("Damage Index", fontsize=8)
        ax.set_yticks(np.linspace(0.0, y_max, 6))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

        xticks = ax.get_xticks()
        xticks = xticks[(xticks >= 0.0) & (xticks <= drift_safe)]
        di_ticks = (pred / drift_safe) * xticks

        ax.plot(
            xticks, di_ticks,
            linestyle="None",
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=5
        )

        ax.scatter(
            [drift_safe],
            [pred],
            s=30,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            zorder=6
        )

        # ---- projection lines to axes ----
        ax.vlines(
            x=drift_safe,
            ymin=0.0,
            ymax=pred,
            colors="black",
            linestyles="dashed",
            linewidth=0.7,
            zorder=4
        )

        ax.hlines(
            y=pred,
            xmin=0.0,
            xmax=drift_safe,
            colors="black",
            linestyles="dashed",
            linewidth=0.7,
            zorder=4
        )

        ax.text(
            drift_safe,
            pred + 0.06,
            f"{pred:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            fontweight="bold",
            zorder=7
        )


        ax.text(1.04, 0.10 / y_max, "ND",  transform=ax.transAxes,
                fontsize=8, color="green", fontweight="bold",
                va="center", ha="left")
        ax.text(1.04, 0.35 / y_max, "PD",  transform=ax.transAxes,
                fontsize=8, color="orange", fontweight="bold",
                va="center", ha="left")
        ax.text(1.04, 0.75 / y_max, "SD",  transform=ax.transAxes,
                fontsize=8, color="red", fontweight="bold",
                va="center", ha="left")
        ax.text(1.04, 1.25 / y_max, "COL", transform=ax.transAxes,
                fontsize=8, color="gray", fontweight="bold",
                va="center", ha="left")

        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        
        ax.spines ["top"].set_linewidth(0.5)
        ax.spines ["right"].set_linewidth(0.5)
        ax.spines ["left"].set_linewidth(0.5)
        ax.spines ["bottom"].set_linewidth(0.5)
        
        ax.tick_params(
    axis="both",
    which="major",
    labelsize=5,
    length=3,
    width=0.5,
    direction="out"
)
        
        ax.grid(True, linestyle="", linewidth=0.5, alpha=0.6)

        plt.tight_layout(pad=0.4)
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
