import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# Fungsi untuk membuat model default jika loading gagal
def create_default_model():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Screen Size', 'refresh_rate']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
         ['Resolution', 'Aspect Ratio'])
    ], remainder='drop')
    
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', SVR(kernel='rbf', C=10, epsilon=5))
    ])

def load_model_safe():
    try:
        # Coba load model baru terlebih dahulu
        try:
            return joblib.load('monitor_price_predictor_new.pkl')
        except:
            # Fallback ke model lama jika baru tidak ada
            return joblib.load('monitor_price_predictor.pkl')
    except Exception as e:
        st.warning(f"Gagal memuat model yang disimpan: {str(e)}. Membuat model default...")
        return create_default_model()

def main():
    st.title("🖥️ Prediksi Harga Monitor (Fixed Version)")
    
    # Load model dengan fallback
    model = load_model_safe()
    
    # Input form
    with st.form("input_form"):
        st.subheader("Spesifikasi Monitor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            screen_size = st.number_input("Ukuran Layar (inch)", min_value=10.0, max_value=100.0, value=24.0, step=0.1)
            refresh_rate = st.number_input("Refresh Rate (Hz)", min_value=60, max_value=360, value=60, step=1)
            
        with col2:
            resolution = st.selectbox("Resolusi", ['FHD', 'QHD', 'UHD', '4K', '5K', '8K', 'HD', 'OLED', 'Others'])
            aspect_ratio = st.selectbox("Aspect Ratio", ['16:9', '21:9', '32:9', '16:10', '4:3'])
        
        submitted = st.form_submit_button("Prediksi Harga")
    
    if submitted:
        input_data = {
            'Screen Size': screen_size,
            'Resolution': resolution,
            'Aspect Ratio': aspect_ratio,
            'refresh_rate': refresh_rate,
            'Brand': 'Unknown'  # placeholder
        }
        
        try:
            input_df = pd.DataFrame([input_data])
            input_df['Screen Size'] = float(input_df['Screen Size'])
            input_df['refresh_rate'] = int(input_df['refresh_rate'])
            
            prediction = model.predict(input_df)
            st.success(f"Perkiraan harga: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error prediksi: {str(e)}")

if __name__ == "__main__":
    main()
