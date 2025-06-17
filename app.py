import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Fungsi untuk memuat model
def load_model():
    try:
        model = joblib.load('monitor_price_predictor.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Fungsi untuk memprediksi harga
def predict_price(model, input_data):
    try:
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Konversi tipe data
        input_df['Screen Size'] = float(input_df['Screen Size'])
        input_df['refresh_rate'] = int(input_df['refresh_rate'])
        
        # Lakukan prediksi
        prediction = model.predict(input_df)
        return round(prediction[0], 2)
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None

# Fungsi utama
def main():
    st.title("🖥️ Prediksi Harga Monitor")
    st.write("""
    Aplikasi ini memprediksi harga monitor berdasarkan spesifikasinya menggunakan model SVR.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file 'monitor_price_predictor.pkl' tersedia.")
        return
    
    # Input form
    with st.form("input_form"):
        st.subheader("Masukkan Spesifikasi Monitor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            screen_size = st.number_input("Ukuran Layar (inch)", min_value=10.0, max_value=100.0, value=24.0, step=0.1)
            refresh_rate = st.number_input("Refresh Rate (Hz)", min_value=60, max_value=360, value=60, step=1)
            
        with col2:
            resolution = st.selectbox("Resolusi", ['FHD', 'QHD', 'UHD', '4K', '5K', '8K', 'HD', 'OLED', 'Others'])
            aspect_ratio = st.selectbox("Aspect Ratio", ['16:9', '21:9', '32:9', '16:10', '4:3'])
        
        submitted = st.form_submit_button("Prediksi Harga")
    
    if submitted:
        # Buat input data
        input_data = {
            'Screen Size': screen_size,
            'Resolution': resolution,
            'Aspect Ratio': aspect_ratio,
            'refresh_rate': refresh_rate,
            'Brand': 'Unknown'  # Kolom ini tidak digunakan tapi diperlukan untuk format input
        }
        
        # Lakukan prediksi
        prediction = predict_price(model, input_data)
        
        if prediction is not None:
            st.success(f"Perkiraan harga monitor: ${prediction:,.2f}")

if __name__ == "__main__":
    main()
