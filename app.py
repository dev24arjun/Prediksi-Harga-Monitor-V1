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
ASPECT_RATIOS = ['Unknown', '4:03', '16:09', '16:10', '17:09', '21:09', '32:09',
                 '1.27:1', '1.38:1', '1.76:1', '1.77:1', '1.78:1', '2.30:1', '2.35:1', '2.40:1']
BRANDS = ['acer', 'Alienware', 'ALOGIC','ANGEL POS','Anmite','AOC', 'AOPEN','ARZOPA','ASUS', 'AUO', 'BenQ', 'Cevaton',
          'BOSII','CIDETTY','cocopar','CRUA','Deco Gear','Dell','DIYmalls','domyfan','Duex','Elo','Fiodio','GIGABYTE',
          'HP','iChawk','INNOCN','InnoView','kasorey','Kensington','KOORUI','KTC','KYY','Lenovo','LESOWN','LG',
          'LILLIPUT','Macsecor','MP','MSI','NEC','Neway','PHILIPS','Philips Computer Monitors','Monitors','Pixio',
          'Planar','Poly','QQH','SAMSUNG','Spectre','SANSUI','SideTrak','Targus','Teamgee','Thermaltake','Tilta',
          'TouchWo','ViewSonic','XGaming','Z Z-EDGE']

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
        # Siapkan input dictionary
        input_data = {
            'Screen Size': screen_size,
            'refresh_rate': refresh_rate
        }

        # Ambil kolom yang digunakan saat training
        dummy_columns = model.feature_names_in_.tolist()

        # Isi semua kolom dengan default 0
        for col in dummy_columns:
            input_data[col] = 0

        # Set nilai dummy aktif berdasarkan input
        res_key = f"Resolution_{resolution}"
        asp_key = f"Aspect Ratio_{aspect_ratio}"
        brand_key = f"Brand_{brand}"

        if res_key in dummy_columns:
            input_data[res_key] = 1
        if asp_key in dummy_columns:
            input_data[asp_key] = 1
        if brand_key in dummy_columns:
            input_data[brand_key] = 1

        # Konversi ke DataFrame
        df_input = pd.DataFrame([input_data])

        try:
            pred = model.predict(df_input)[0]
            formatted_price = f"${pred:,.2f}"
            st.success(f"💰 Prediksi harga monitor: **{formatted_price}**")

            # Simpan ke riwayat sesi
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

