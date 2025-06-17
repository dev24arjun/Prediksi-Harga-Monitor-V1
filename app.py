import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Monitor", layout="wide")
st.title("🖥️ Prediksi Harga Monitor")

# Fungsi untuk load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('monitor_price_predictor_new.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()  # Hentikan aplikasi jika model tidak bisa dimuat

# Load model
model = load_model()

# Form input
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        screen_size = st.number_input("Ukuran Layar (inch)", 
                                    min_value=10.0, 
                                    max_value=100.0, 
                                    value=24.0, 
                                    step=0.1)
        refresh_rate = st.number_input("Refresh Rate (Hz)", 
                                     min_value=60, 
                                     max_value=360, 
                                     value=144)
    
    with col2:
        resolution = st.selectbox("Resolusi", 
                                ['FHD', 'QHD', 'UHD', '4K', '5K', '8K', 'HD', 'OLED', 'Others'])
        aspect_ratio = st.selectbox("Aspect Ratio", 
                                  ['16:9', '21:9', '32:9', '16:10', '4:3'])
    
    submitted = st.form_submit_button("Prediksi Harga")

# Proses prediksi
if submitted:
    input_data = {
        'Screen Size': screen_size,
        'Resolution': resolution,
        'Aspect Ratio': aspect_ratio,
        'refresh_rate': refresh_rate,
        'Brand': 'Unknown'  # Kolom dummy
    }
    
    try:
        # Konversi ke DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Pastikan tipe data sesuai
        input_df['Screen Size'] = input_df['Screen Size'].astype(float)
        input_df['refresh_rate'] = input_df['refresh_rate'].astype(int)
        
        # Prediksi
        prediction = model.predict(input_df)[0]
        
        # Tampilkan hasil
        st.success(f"Perkiraan harga monitor: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
