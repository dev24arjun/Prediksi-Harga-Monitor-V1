import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load nilai unik dari dataset untuk dropdown ===
BRANDS = ['Others', 'ALOGIC', 'ANGEL POS', 'AOC', 'AOPEN', 'ARZOPA', 'ASUS', 'AUO', 'Alienware', 'Anmite', 'BOSII']
RESOLUTIONS = ['Others', '2K DCI 1080p', '3000', '3440 x 1440 (UWQHD)', '3840 x 2160 (UHD)', '3840 x 2160 UHD',
               '480 x 272', '4K', '4K DCI 2160p', '4K HDR 2016', '4K UHD']
ASPECT_RATIOS = ['Others', '1.27:1', '1.38:1', '1.76:1', '1.77:1', '1.78:1', '16:09', '16:10', '17:09', '2.30:1', '2.35:1']

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Monitor", layout="centered")
st.title("🖥️ Prediksi Harga Monitor")
st.markdown("Masukkan spesifikasi monitor untuk memprediksi harga menggunakan model SVR.")

@st.cache_resource
def load_model():
    try:
        return joblib.load("monitor_price_predictor_new.pkl")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

model = load_model()

# Inisialisasi sesi prediksi
if 'history' not in st.session_state:
    st.session_state.history = []

# Form input pengguna
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

        # Semua kolom dummy yang digunakan saat training
        dummy_columns = [
            'Resolution_2K DCI 1080p', 'Resolution_3000', 'Resolution_3440 x 1440 (UWQHD)',
            'Resolution_3840 x 2160 (UHD)', 'Resolution_3840 x 2160 UHD', 'Resolution_480 x 272',
            'Resolution_4K', 'Resolution_4K DCI 2160p', 'Resolution_4K HDR 2016', 'Resolution_4K UHD',
            'Aspect Ratio_1.27:1', 'Aspect Ratio_1.38:1', 'Aspect Ratio_1.76:1', 'Aspect Ratio_1.77:1',
            'Aspect Ratio_1.78:1', 'Aspect Ratio_16:09', 'Aspect Ratio_16:10', 'Aspect Ratio_17:09',
            'Aspect Ratio_2.30:1', 'Aspect Ratio_2.35:1',
            'Brand_ALOGIC', 'Brand_ANGEL POS', 'Brand_AOC', 'Brand_AOPEN', 'Brand_ARZOPA', 'Brand_ASUS',
            'Brand_AUO', 'Brand_Alienware', 'Brand_Anmite', 'Brand_BOSII'
        ]

        for col in dummy_columns:
            input_data[col] = 0

        # Cek dan aktifkan dummy input jika cocok
        res_col = f'Resolution_{resolution}' if resolution != 'Others' else None
        asp_col = f'Aspect Ratio_{aspect_ratio}' if aspect_ratio != 'Others' else None
        brand_col = f'Brand_{brand}' if brand != 'Others' else None

        if res_col in dummy_columns:
            input_data[res_col] = 1
        if asp_col in dummy_columns:
            input_data[asp_col] = 1
        if brand_col in dummy_columns:
            input_data[brand_col] = 1

        df_input = pd.DataFrame([input_data])

        try:
            pred = model.predict(df_input)[0]
            formatted_price = f"${pred:,.2f}"
            st.success(f"💰 Prediksi harga monitor: **{formatted_price}**")

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

# Tampilkan riwayat sesi
if st.session_state.history:
    st.markdown("### 🧾 Riwayat Prediksi")
    df_hist = pd.DataFrame(st.session_state.history)
    df_hist['Prediksi'] = df_hist['Prediksi'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(df_hist, use_container_width=True)

    st.markdown("### 📊 Grafik Prediksi Harga")
    chart_df = pd.DataFrame(st.session_state.history)[['Prediksi']]
    st.bar_chart(chart_df)
