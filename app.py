import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("encoder.pkl")

st.title("💧 Water Quality Prediction System")

# Inputs
pH = st.number_input("pH", 0.0, 14.0, 7.0)
DO = st.number_input("DO", 0.0, 14.0, 6.0)
BOD = st.number_input("BOD", 0.0, 20.0, 2.0)
conductivity = st.number_input("Conductivity", 0.0, 2000.0, 500.0)
nitrate = st.number_input("Nitrate", 0.0, 50.0, 10.0)
fecal_coliform = st.number_input("Fecal Coliform", 0.0, 10000.0, 100.0)
total_coliform = st.number_input("Total Coliform", 0.0, 10000.0, 200.0)

if st.button("Predict"):

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

    if result == "Unsafe":
        st.error(f"⚠️ Water Quality: {result}")
    else:
        st.success(f"✅ Water Quality: {result}")