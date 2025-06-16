import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and supporting files
model = joblib.load('monitor_price_predictor.pkl')
feature_columns = joblib.load('feature_columns.pkl')
categories = joblib.load('categories.pkl')

st.title('Monitor Price Predictor')
st.write('Predict the price of a monitor based on its specifications')

# Create input fields
screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=100.0, value=24.0, step=0.1)
refresh_rate = st.number_input('Refresh Rate (Hz)', min_value=60, max_value=360, value=60, step=1)
resolution = st.selectbox('Resolution', sorted(categories['Resolution']))
aspect_ratio = st.selectbox('Aspect Ratio', sorted(categories['Aspect Ratio']))

# Create a button to make prediction
if st.button('Predict Price'):
    # Create input dataframe
    input_data = pd.DataFrame([[screen_size, resolution, aspect_ratio, 'Unknown', refresh_rate]], 
                             columns=feature_columns)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f'Predicted Price: ${prediction:,.2f}')
    
    # Show some examples from training data (optional)
    st.subheader('Sample Monitors with Similar Specs')
    # You could add code here to show similar monitors from your dataset