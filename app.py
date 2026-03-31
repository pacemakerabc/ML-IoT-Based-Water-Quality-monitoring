import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="💧",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("encoder.pkl")

# -----------------------------
# CUSTOM STYLE (🔥 BEAUTIFUL UI)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #0077b6;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="title">💧 Water Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Water Quality Classification using WQI & ML</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# INPUT SECTION (2 COLUMNS)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    pH = st.slider("pH", 0.0, 14.0, 7.0)
    DO = st.slider("Dissolved Oxygen (DO)", 0.0, 14.0, 6.0)
    BOD = st.slider("BOD", 0.0, 20.0, 2.0)
    conductivity = st.slider("Conductivity", 0.0, 2000.0, 500.0)

with col2:
    nitrate = st.slider("Nitrate", 0.0, 50.0, 10.0)
    fecal_coliform = st.slider("Fecal Coliform", 0.0, 10000.0, 100.0)
    total_coliform = st.slider("Total Coliform", 0.0, 10000.0, 200.0)

st.markdown("---")

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict Water Quality"):

    DO = DO if DO != 0 else 0.1

    pollution_index = BOD / DO
    coliform_load = fecal_coliform + total_coliform
    pH_dev = abs(pH - 7)
    nutrient_load = nitrate + BOD

    data = pd.DataFrame([{
        'DO': DO,
        'pH': pH,
        'Conductivity': conductivity,
        'BOD': BOD,
        'Nitrate': nitrate,
        'Fecal_Coliform': fecal_coliform,
        'Total_Coliform': total_coliform,
        'Pollution_Index': pollution_index,
        'Coliform_Load': coliform_load,
        'pH_Deviation': pH_dev,
        'Nutrient_Load': nutrient_load
    }])

    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    result = le.inverse_transform(pred)[0]

    st.markdown("---")

    # -----------------------------
    # RESULT DISPLAY (🔥 STYLISH)
    # -----------------------------
    if result == "A":
        st.success("🟢 Class A - Drinking Water Quality (Excellent)")
    elif result == "B":
        st.success("🟢 Class B - Bathing Quality (Good)")
    elif result == "C":
        st.warning("🟡 Class C - Requires Treatment")
    elif result == "D":
        st.warning("🟠 Class D - Suitable for Fisheries")
    else:
        st.error("🔴 Class E - Highly Polluted")

    # -----------------------------
    # SHOW INPUT SUMMARY
    # -----------------------------
    st.markdown("### 📊 Input Summary")
    st.write(data)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🔬 Developed using Machine Learning & WQI-based Classification")