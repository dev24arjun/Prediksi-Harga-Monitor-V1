import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and categories
pipeline = joblib.load('monitor_price_predictor.pkl')
categories = joblib.load('categories.pkl')

st.title('Monitor Price Predictor')
st.write("""
This app predicts the price of monitors based on their specifications using a Support Vector Regression (SVR) model.
""")

# Sidebar with user inputs
st.sidebar.header('User Input Features')

def user_input_features():
    screen_size = st.sidebar.slider('Screen Size (inches)', 15.0, 50.0, 24.0, 0.1)
    refresh_rate = st.sidebar.slider('Refresh Rate (Hz)', 60, 360, 144, 1)
    
    resolution = st.sidebar.selectbox('Resolution', categories['Resolution'])
    aspect_ratio = st.sidebar.selectbox('Aspect Ratio', categories['Aspect Ratio'])
    
    data = {
        'Screen Size': screen_size,
        'refresh_rate': refresh_rate,
        'Resolution': resolution,
        'Aspect Ratio': aspect_ratio
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input')
st.write(input_df)

# Make prediction
prediction = pipeline.predict(input_df)

st.subheader('Prediction')
st.write(f'Predicted Price: ${prediction[0]:,.2f}')

# Add some information about the model
st.subheader('Model Information')
st.write("""
- **Model Type**: Support Vector Regression (SVR)
- **Parameters**: 
  - Kernel: rbf
  - C: 10
  - Epsilon: 5
""")

# Add some sample data for reference
st.subheader('Common Monitor Specifications for Reference')
st.write("""
| Type | Screen Size | Resolution | Refresh Rate | Typical Price Range |
|------|-------------|------------|--------------|---------------------|
| Budget | 21-24" | FHD | 60-75Hz | $100-$200 |
| Mainstream | 24-27" | QHD | 144-165Hz | $250-$500 |
| Premium | 27-32" | 4K | 60-144Hz | $500-$1000 |
| Gaming | 24-27" | FHD/QHD | 144-240Hz | $300-$800 |
| Ultra-wide | 34-38" | UWQHD | 100-144Hz | $700-$1500 |
""")
