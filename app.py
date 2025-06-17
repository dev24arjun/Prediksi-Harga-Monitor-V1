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
ASPECT_RATIOS = ['Unknown', '4:03', '16:09', '16:10', '17:09', '21:09', '32:09', '1.27:1', '1.38:1', '1.76:1', '1.77:1', '1.78:1', '2.30:1', '2.35:1', '2.40:1']
BRANDS = ['acer', 'Alienware', 'ALOGIC', 'ANGEL POS', 'Anmite', 'AOC', 'AOPEN', 'ARZOPA', 'ASUS', 'AUO', 'BenQ', 'Cevaton', 'BOSII', 'CIDETTY', 'cocopar', 'CRUA',
          'Deco Gear', 'Dell', 'DIYmalls', 'domyfan', 'Duex', 'Elo', 'Fiodio', 'GIGABYTE', 'HP', 'iChawk', 'INNOCN', 'InnoView', 'kasorey', 'Kensington',
          'KOORUI', 'KTC', 'KYY', 'Lenovo', 'LESOWN', 'LG', 'LILLIPUT', 'Macsecor', 'MP', 'MSI', 'NEC', 'Neway', 'PHILIPS', 'Philips Computer Monitors',
          'Monitors', 'Pixio', 'Planar', 'Poly', 'QQH', 'SAMSUNG', 'Spectre', 'SANSUI', 'SideTrak', 'Targus', 'Teamgee', 'Thermaltake', 'Tilta', 'TouchWo',
          'ViewSonic', 'XGaming', 'Z Z-EDGE']

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

        dummy_columns = [
            # ==== Resolution ====
            'Resolution_2K', 'Resolution_3K', 'Resolution_4K', 'Resolution_5K', 'Resolution_8K',
            'Resolution_FHD', 'Resolution_HD', 'Resolution_OLED', 'Resolution_QHD', 'Resolution_UHD',

            # ==== Aspect Ratio ====
            'Aspect Ratio_Unknown', 'Aspect Ratio_4:03', 'Aspect Ratio_16:09', 'Aspect Ratio_16:10',
            'Aspect Ratio_17:09', 'Aspect Ratio_21:09', 'Aspect Ratio_32:09',
            'Aspect Ratio_1.27:1', 'Aspect Ratio_1.38:1', 'Aspect Ratio_1.76:1',
            'Aspect Ratio_1.77:1', 'Aspect Ratio_1.78:1', 'Aspect Ratio_2.30:1',
            'Aspect Ratio_2.35:1', 'Aspect Ratio_2.40:1',

            # ==== Brand ====
            'Brand_acer', 'Brand_Alienware', 'Brand_ALOGIC', 'Brand_ANGEL POS', 'Brand_Anmite',
            'Brand_AOC', 'Brand_AOPEN', 'Brand_ARZOPA', 'Brand_ASUS', 'Brand_AUO',
            'Brand_BenQ', 'Brand_Cevaton', 'Brand_BOSII', 'Brand_CIDETTY', 'Brand_cocopar',
            'Brand_CRUA', 'Brand_Deco Gear', 'Brand_Dell', 'Brand_DIYmalls', 'Brand_domyfan',
            'Brand_Duex', 'Brand_Elo', 'Brand_Fiodio', 'Brand_GIGABYTE', 'Brand_HP',
            'Brand_iChawk', 'Brand_INNOCN', 'Brand_InnoView', 'Brand_kasorey', 'Brand_Kensington',
            'Brand_KOORUI', 'Brand_KTC', 'Brand_KYY', 'Brand_Lenovo', 'Brand_LESOWN',
            'Brand_LG', 'Brand_LILLIPUT', 'Brand_Macsecor', 'Brand_MP', 'Brand_MSI',
            'Brand_NEC', 'Brand_Neway', 'Brand_PHILIPS', 'Brand_Philips Computer Monitors',
            'Brand_Monitors', 'Brand_Pixio', 'Brand_Planar', 'Brand_Poly', 'Brand_QQH',
            'Brand_SAMSUNG', 'Brand_Spectre', 'Brand_SANSUI', 'Brand_SideTrak', 'Brand_Targus',
            'Brand_Teamgee', 'Brand_Thermaltake', 'Brand_Tilta', 'Brand_TouchWo', 'Brand_ViewSonic',
            'Brand_XGaming', 'Brand_Z Z-EDGE'
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
