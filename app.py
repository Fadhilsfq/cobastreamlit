import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# ==========================================
# Konfigurasi Halaman Dashboard
# ==========================================
st.set_page_config(page_title="Dashboard Prediksi Depresi", layout="wide")

st.title("Dashboard Prediksi Depresi Mahasiswa")
st.markdown("Implementasi model CatBoost dengan hyperparameter optimal dari Optuna.")
st.markdown("---")

# Membuat dua tab terpisah
tab1, tab2 = st.tabs(["Deteksi Teks (Sistem Aturan)", "Prediksi Model (CatBoost)"])

# ==========================================
# TAB 1: Fitur Input Teks (Tanpa NLP / Rule-Based)
# ==========================================
with tab1:
    st.header("Cek Kondisi Mental dari Teks")
    st.write("Fitur ini mengevaluasi teks murni menggunakan pencocokan kamus kata (Rule-Based), tanpa algoritma AI/NLP tambahan.")
    
    user_input = st.text_area("Ceritakan apa yang sedang Anda rasakan saat ini:")

    if st.button("Hitung Persentase", key="btn_teks"):
        if user_input:
            text_lower = user_input.lower()
            
            # Sistem Kamus Kata (Lexicon)
            high_risk = ['bunuh diri', 'mati', 'akhiri hidup', 'tidak berguna', 'putus asa']
            medium_risk = ['capek', 'lelah', 'stres', 'depresi', 'sedih', 'hancur', 'menyerah']
            
            # Logika perhitungan skor murni
            score = 5 
            if any(word in text_lower for word in high_risk):
                score += 85
            elif any(word in text_lower for word in medium_risk):
                score += 45
            else:
                score += 15
                
            score = min(score, 98) # Maksimal 98%
            
            st.metric(label="Indikasi Depresi (Berdasarkan Teks)", value=f"{score}%")
            
            if score >= 80:
                st.error("Risiko Tinggi. Tolong jangan ragu untuk mencari bantuan profesional.")
            elif score >= 40:
                st.warning("Risiko Sedang. Pertimbangkan untuk beristirahat atau bercerita kepada seseorang.")
            else:
                st.success("Risiko Rendah. Emosi wajar, tetap jaga kesehatan mental Anda.")
        else:
            st.warning("Ketikkan teks terlebih dahulu.")

# ==========================================
# TAB 2: Prediksi dengan CatBoost + Optuna
# ==========================================
with tab2:
    st.header("Prediksi Depresi Mahasiswa")
    st.write("Masukkan data akademik dan kebiasaan mahasiswa untuk diprediksi menggunakan model CatBoost.")
    
    # Form Input untuk fitur numerik dan kategorikal
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Umur (Age)", min_value=15, max_value=60, value=20)
        academic_pressure = st.slider("Tekanan Akademik (0-5)", 0.0, 5.0, 3.0, 0.5)
        cgpa = st.number_input("IPK (CGPA)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        study_satisfaction = st.slider("Kepuasan Belajar (0-5)", 0.0, 5.0, 3.0, 0.5)
        financial_stress = st.slider("Stres Finansial (1-5)", 1.0, 5.0, 3.0, 0.5)
        
    with col2:
        sleep_duration = st.selectbox("Durasi Tidur", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Pola Makan", ["Healthy", "Moderate", "Unhealthy"])
        suicidal_thoughts = st.selectbox("Pernah Memikirkan Bunuh Diri?", ["Yes", "No"])
        family_history = st.selectbox("Riwayat Penyakit Mental Keluarga", ["Yes", "No"])
        work_study_hours = st.number_input("Jam Kerja/Belajar per Hari", min_value=0.0, max_value=24.0, value=8.0)

    if st.button("Prediksi dengan CatBoost", key="btn_catboost"):
        try:
            # 1. Load Model CatBoost yang sudah ditraining
            model = CatBoostClassifier()
            model.load_model('catboost_optuna_model.cbm')
            
            # 2. Siapkan data dari input pengguna ke dalam format DataFrame
            # Pastikan nama kolom sama persis dengan yang digunakan saat training di notebook
            input_data = pd.DataFrame({
                'Age': [age],
                'Academic Pressure': [academic_pressure],
                'CGPA': [cgpa],
                'Study Satisfaction': [study_satisfaction],
                'Sleep Duration': [sleep_duration],
                'Dietary Habits': [dietary_habits],
                'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
                'Work/Study Hours': [work_study_hours],
                'Financial Stress': [financial_stress],
                'Family History of Mental Illness': [family_history]
                # Tambahkan kolom lain di sini jika model Anda menggunakan fitur tambahan dari dataset
            })
            
            # 3. Lakukan prediksi
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[0]
            
            st.markdown("### Hasil Prediksi")
            if prediction[0] == 1:
                st.error(f"**Terindikasi Depresi** (Probabilitas: {prediction_proba[1]*100:.2f}%)")
            else:
                st.success(f"**Tidak Terindikasi Depresi** (Probabilitas: {prediction_proba[0]*100:.2f}%)")
                
        except Exception as e:
            st.error("Model 'catboost_optuna_model.cbm' belum ditemukan. Pastikan Anda telah menyimpan model CatBoost dari Notebook Anda ke dalam format .cbm di folder yang sama.")
