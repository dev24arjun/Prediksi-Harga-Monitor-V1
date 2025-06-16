import streamlit as st
import joblib
import pandas as pd
import sklearn
from sklearn.svm import SVR

# Ensure consistent config
sklearn.set_config(transform_output="pandas")

# Configure app
st.set_page_config(page_title="Monitor Price Predictor", layout="centered")

@st.cache_resource
def load_model():
    try:
        # Load the fixed model
        model = joblib.load('monitor_price_predictor_fixed.pkl')
        cats = joblib.load('categories_fixed.pkl')
        return model, cats
    except Exception as e:
        st.error(f"Model loading error. Please ensure you're using scikit-learn 1.3+")
        st.stop()

pipeline, categories = load_model()

# App UI
st.title("🖥️ Monitor Price Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        screen_size = st.slider("Screen Size (inches)", 15.0, 50.0, 24.0, 0.1)
        refresh_rate = st.slider("Refresh Rate (Hz)", 60, 360, 144)
    
    with col2:
        resolution = st.selectbox("Resolution", options=categories['Resolution'])
        aspect_ratio = st.selectbox("Aspect Ratio", options=categories['Aspect Ratio'])
    
    if st.form_submit_button("Predict Price"):
        input_data = pd.DataFrame({
            'Screen Size': [screen_size],
            'refresh_rate': [refresh_rate],
            'Resolution': [resolution],
            'Aspect Ratio': [aspect_ratio]
        })
        
        try:
            price = pipeline.predict(input_data)[0]
            st.success(f"Predicted Price: ${price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Add requirements note
st.sidebar.markdown("""
**Requirements:**
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
""")
