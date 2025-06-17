import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==== Konfigurasi halaman ====
st.set_page_config(page_title="Prediksi Harga Monitor", layout="centered")
st.title("🖥️ Prediksi Harga Monitor")
st.markdown("Masukkan spesifikasi monitor untuk memprediksi harga menggunakan model SVR.")

# ==== Load model ====
@st.cache_resource
def load_model():
    try:
        return joblib.load("monitor_price_predictor_new.pkl")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

model = load_model()

# ==== Inisialisasi riwayat sesi ====
if 'history' not in st.session_state:
    st.session_state.history = []

# ==== Nilai dropdown ====
RESOLUTIONS = ['Others', '2K', '3K', '4K', '5K', '8K', 'FHD', 'HD', 'OLED', 'QHD', 'UHD']
ASPECT_RATIOS = ['Others', '1.27:1', '1.38:1', '1.76:1', '1.77:1', '1.78:1', '16:09', '16:10', '17:09', '2.30:1', '2.35:1']
BRANDS = ['Others', 'ALOGIC', 'ANGEL POS', 'AOC', 'AOPEN', 'ARZOPA', 'ASUS', 'AUO', 'Alienware', 'Anmite', 'BOSII']

# ==== Form input ====
with st.form("form_prediksi"):
    st.subheader("📥 Masukkan Spesifikasi Monitor")

    screen_size = st.number_input("Ukuran Layar (inch)", min_value=10.0, max_value=60.0, step=0.1, value=24.0)
    refresh_rate = st.number_input("Refresh Rate (Hz)", min_value=60, max_value=360, step=1, value=75)
    resolution = st.selectbox("Resolusi", RESOLUTIONS)
    aspect_ratio = st.selectbox("Aspect Ratio", ASPECT_RATIOS)
    brand = st.selectbox("Merek", BRANDS)

    submitted = st.form_submit_button("Prediksi Harga")

    if submitted:
        # Template input
        input_data = {
            'Screen Size': screen_size,
            'refresh_rate': refresh_rate
        }

        # Dummy kolom (harus sesuai hasil get_dummies dari training)
        dummy_columns = [
            'Resolution_2K', 'Resolution_3K', 'Resolution_4K', 'Resolution_5K', 'Resolution_8K',
            'Resolution_FHD', 'Resolution_HD', 'Resolution_OLED', 'Resolution_QHD', 'Resolution_UHD',
            'Aspect Ratio_1.27:1', 'Aspect Ratio_1.38:1', 'Aspect Ratio_1.76:1', 'Aspect Ratio_1.77:1',
            'Aspect Ratio_1.78:1', 'Aspect Ratio_16:09', 'Aspect Ratio_16:10', 'Aspect Ratio_17:09',
            'Aspect Ratio_2.30:1', 'Aspect Ratio_2.35:1',
            'Brand_ALOGIC', 'Brand_ANGEL POS', 'Brand_AOC', 'Brand_AOPEN', 'Brand_ARZOPA', 'Brand_ASUS',
            'Brand_AUO', 'Brand_Alienware', 'Brand_Anmite', 'Brand_BOSII'
        ]

        for col in dummy_columns:
            input_data[col] = 0

        # Set nilai dummy sesuai input
        if f'Resolution_{resolution}' in dummy_columns:
            input_data[f'Resolution_{resolution}'] = 1
        if f'Aspect Ratio_{aspect_ratio}' in dummy_columns:
            input_data[f'Aspect Ratio_{aspect_ratio}'] = 1
        if f'Brand_{brand}' in dummy_columns:
            input_data[f'Brand_{brand}'] = 1

        df_input = pd.DataFrame([input_data])

        try:
            pred = model.predict(df_input)[0]
            formatted_price = f"${pred:,.2f}"
            st.success(f"💰 Prediksi harga monitor: **{formatted_price}**")

            # Simpan ke sesi
            st.session_state.history.append({
                "Ukuran": screen_size,
                "Refresh Rate": refresh_rate,
                "Resolusi": resolution,
                "Rasio": aspect_ratio,
                "Merek": brand,
                "Prediksi": pred
            })

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")

# ==== Tampilkan riwayat prediksi ====
if st.session_state.history:
    st.markdown("### 🧾 Riwayat Prediksi")
    df_hist = pd.DataFrame(st.session_state.history)
    df_hist['Prediksi'] = df_hist['Prediksi'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(df_hist, use_container_width=True)

    st.markdown("### 📊 Grafik Prediksi Harga")
    chart_df = pd.DataFrame(st.session_state.history)[['Prediksi']]
    st.bar_chart(chart_df)
