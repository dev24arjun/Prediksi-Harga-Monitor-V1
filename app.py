import streamlit as st
import joblib
import pandas as pd

# Load model and metadata
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('monitor_price_predictor.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        categories = joblib.load('categories.pkl')
        return model, feature_columns, categories
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        return None, None, None

model, feature_columns, categories = load_assets()

if model is None:
    st.stop()

# App UI
st.title('Monitor Price Predictor')
st.markdown("""
Predict the price of a monitor based on its specifications.
The model uses SVR with RBF kernel (C=10, epsilon=5).
""")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        screen_size = st.number_input(
            'Screen Size (inches)',
            min_value=10.0,
            max_value=100.0,
            value=24.0,
            step=0.1
        )
        refresh_rate = st.number_input(
            'Refresh Rate (Hz)',
            min_value=60,
            max_value=360,
            value=144,
            step=1
        )
    
    with col2:
        resolution = st.selectbox(
            'Resolution',
            categories['Resolution']
        )
        aspect_ratio = st.selectbox(
            'Aspect Ratio',
            categories['Aspect Ratio']
        )
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        # Create input dataframe with correct column order
        input_data = pd.DataFrame([[
            screen_size,
            resolution,
            aspect_ratio,
            'Unknown',  # Brand (as placeholder)
            refresh_rate
        ]], columns=feature_columns['all_columns'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success(f'**Predicted Price:** ${prediction:,.2f}')
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Model information sidebar
st.sidebar.markdown("""
### Model Information
- **Algorithm:** Support Vector Regression (SVR)
- **Kernel:** RBF
- **Parameters:**
  - C = 10
  - epsilon = 5
- **Features Used:**
  - Screen Size (numeric)
  - Resolution (categorical)
  - Aspect Ratio (categorical)
  - Refresh Rate (numeric)
""")
