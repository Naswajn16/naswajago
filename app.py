import streamlit as st
import joblib
import numpy as np

model_nb = joblib.load('model_heart_disease.pkl')
scaler = joblib.load('scaler_heart.pkl') 

st.title("Prediksi Penyakit Jantung RS Sehat Sentosa")
st.write("Silakan masukkan data medis pasien di bawah ini:")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia", min_value=1, max_value=100)
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600)
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250)
   
    with col2:
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=Yes, 0=No)", [0, 1])
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", [0, 1])

    
    submit_button = st.form_submit_button("🩺 Prediksi Sekarang")

if submit_button:
   
    input_data = np.array([[age, cp, trestbps, chol, fbs, restecg, thalach, exang]])

    input_scaled = scaler.transform(input_data)
    
    hasil = model_nb.predict(input_scaled)
    
    if hasil[0] == 1:
        st.error("Peringatan: Pasien Terindikasi Memiliki Penyakit Jantung.")
    else:
        st.success("Aman: Pasien Terindikasi Tidak Memiliki Penyakit Jantung.")
