import streamlit as st
import joblib
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure page
st.set_page_config(
    page_title="Monitor Price Prediction",
    page_icon="🖥️",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .prediction-box {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stSlider > div > div > div > div {
        background: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Cache resources
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('monitor_price_predictor_v2.pkl')
        cats = joblib.load('categories_v2.pkl')
        return model, cats
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

pipeline, categories = load_assets()

# Main app
st.title("🖥️ Monitor Price Prediction")
st.write("Predict monitor prices using our SVR model with optimized parameters")

# Input section
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        screen_size = st.slider(
            "Screen Size (inches)",
            min_value=15.0,
            max_value=50.0,
            value=24.0,
            step=0.1
        )
        refresh_rate = st.slider(
            "Refresh Rate (Hz)",
            min_value=60,
            max_value=360,
            value=144,
            step=1
        )
    
    with col2:
        resolution = st.selectbox(
            "Resolution",
            options=categories['Resolution'],
            index=categories['Resolution'].index('FHD') if 'FHD' in categories['Resolution'] else 0
        )
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            options=categories['Aspect Ratio'],
            index=0
        )
    
    submitted = st.form_submit_button("Predict Price")

# Prediction logic
if submitted:
    input_data = pd.DataFrame({
        'Screen Size': [screen_size],
        'refresh_rate': [refresh_rate],
        'Resolution': [resolution],
        'Aspect Ratio': [aspect_ratio]
    })
    
    try:
        prediction = pipeline.predict(input_data)[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Predicted Price:</h3>
            <h2>${prediction:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model details
        with st.expander("Model Specifications"):
            st.markdown("""
            **Model Architecture:**
            - Type: Support Vector Regression (SVR)
            - Kernel: rbf
            - C: 10
            - Epsilon: 5
            
            **Preprocessing:**
            - Numerical Features (StandardScaler):
              - Screen Size
              - Refresh Rate
            - Categorical Features (OneHotEncoder):
              - Resolution
              - Aspect Ratio
            """)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Example predictions
st.divider()
st.subheader("Common Configurations")

examples = [
    {"label": "Budget 1080p", "size": 24.0, "res": "FHD", "aspect": "16:9", "refresh": 75},
    {"label": "Gaming Monitor", "size": 27.0, "res": "QHD", "aspect": "16:9", "refresh": 165},
    {"label": "4K Professional", "size": 32.0, "res": "4K", "aspect": "16:9", "refresh": 60},
]

cols = st.columns(len(examples))
for idx, example in enumerate(examples):
    with cols[idx]:
        st.markdown(f"**{example['label']}**")
        st.write(f"Screen: {example['size']}″")
        st.write(f"Res: {example['res']}")
        st.write(f"Refresh: {example['refresh']}Hz")
        
        if st.button(f"Predict {example['label']}", key=f"ex_{idx}"):
            ex_data = pd.DataFrame({
                'Screen Size': [example['size']],
                'Resolution': [example['res']],
                'Aspect Ratio': [example['aspect']],
                'refresh_rate': [example['refresh']]
            })
            ex_pred = pipeline.predict(ex_data)[0]
            st.success(f"${ex_pred:,.2f}")
