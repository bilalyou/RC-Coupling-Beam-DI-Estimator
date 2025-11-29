# RC Coupling Beam Damage Index Estimator

A Streamlit-based graphical user interface (GUI) for **machine-learning–driven estimation of the Park–Ang Damage Index** of reinforced concrete coupling beams.

The app integrates several optimized ML models and a practical equation-based solution, trained on a **comprehensive database of 200 RC coupling beams** and validated against full-scale experimental tests.

---

## 🌟 Features

- ML models included:
  - CatBoost
  - XGBoost  
  - LightGBM  
  - ExtraTrees  
  - MLP  
  - Practical Solution (PS) equation
- User-friendly **Streamlit** web interface
- Input panel for geometry, material strengths, and reinforcement ratios
- Instant prediction of:
  - Damage Index (DI)
  - Damage state classification
  - Drift–DI plot
- Export of prediction results as CSV

---

## 🔗 Live App

Access the web app here:

> **Streamlit app:** _(add your Streamlit URL here)_

---

## 🧱 Background

This tool accompanies the research:

> **“ML-Based Damage Index Estimation of RC Coupling Beams: Data Establishment, Solution Development, Experimental Validation” – Bilal Younis (2025)**

The underlying framework is based on:

- A curated dataset of **200** conventionally and diagonally reinforced RC coupling beams  
- Bayesian-optimized ML models (CatBoost, XGBoost, LightGBM, ExtraTrees, MLP)  
- A minimalistic MLP-based **explicit equation** (PS) for engineering use  
- Blind validation using **full-scale experiments** on RC coupling beams with different failure modes

---

## ⚙️ Installation (Local Use)

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/RC-Coupling-Beam-DI-Estimator.git
cd RC-Coupling-Beam-DI-Estimator
pip install -r requirements.txt
