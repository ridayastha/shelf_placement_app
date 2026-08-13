import streamlit as st
import pandas as pd
import joblib

# Set page title and layout
st.set_page_config(
    page_title="Shelf Placement Predictor",
    page_icon="🛒",
    layout="wide"
)

# Load the trained model and scaler (cached for fast execution)
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Ensure 'model.pkl' and 'scaler.pkl' are in the same directory as app.py.")
    st.stop()

# Title and Description
st.title("🛒 Shelf Placement Movement Predictor")
st.write("Adjust product and shelf placement parameters below to predict product turnover class (**Fast**, **Ignored**, or **Slow**).")

st.divider()

# Organize inputs into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location & Category Settings")
    zone_code = st.number_input("Zone Code", min_value=0, max_value=10, value=0, step=1)
    height_code = st.selectbox("Shelf Height Code", options=[0, 1, 2], help="0: Low, 1: Eye Level, 2: High")
    item_category_code = st.number_input("Item Category Code", min_value=0, max_value=20, value=0, step=1)
    city_code = st.number_input("City Code", min_value=0, max_value=10, value=0, step=1)

with col2:
    st.subheader("📊 Sales & Traffic Metrics")
    price = st.number_input("Price ($)", min_value=0.0, value=26.0, step=0.50)
    discount_percent = st.slider("Discount Percent (%)", min_value=0.0, max_value=100.0, value=14.0)
    daily_customer_traffic = st.number_input("Daily Customer Traffic", min_value=0.0, value=20.0, step=5.0)
    nearby_promotion = st.radio("Nearby Promotion Available?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# Prediction Logic
if st.button("🚀 Predict Product Movement", use_container_width=True, type="primary"):
    
    # 1. Store input dictionary matching EXACT Colab column order
    raw_inputs = {
        'zone_code': zone_code,
        'height_code': height_code,
        'item_category_code': item_category_code,
        'price': price,
        'discount_percent': discount_percent,
        'daily_customer_traffic': daily_customer_traffic,  # 6th position
        'nearby_promotion': nearby_promotion,              # 7th position
        'city_code': city_code
    }

    # 2. Reorder columns if feature names exist in scaler/model attributes
    if hasattr(scaler, 'feature_names_in_'):
        expected_columns = list(scaler.feature_names_in_)
        input_df = pd.DataFrame([raw_inputs])[expected_columns]
    elif hasattr(model, 'feature_names_in_'):
        expected_columns = list(model.feature_names_in_)
        input_df = pd.DataFrame([raw_inputs])[expected_columns]
    else:
        # Fallback: enforce exact order list
        col_order = ['zone_code', 'height_code', 'item_category_code', 'price', 
                     'discount_percent', 'daily_customer_traffic', 'nearby_promotion', 'city_code']
        input_df = pd.DataFrame([raw_inputs])[col_order]

    # Debug expander to verify passed columns
    with st.expander("🔍 Debug: Inspect Passed Features & Column Order"):
        st.write("**DataFrame Sent to Scaler:**", input_df)

    # 3. Scale input features
    scaled_features = scaler.transform(input_df)

    # 4. Predict raw label and probabilities
    raw_prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    # Map raw predictions safely (handles string or numeric types)
    prediction_str = str(raw_prediction).strip().lower()
    
    if prediction_str in ['0', 'fast']:
        label_text = "Fast ⚡"
        status_type = "success"
    elif prediction_str in ['1', 'ignored']:
        label_text = "Ignored 💤"
        status_type = "error"
    elif prediction_str in ['2', 'slow']:
        label_text = "Slow 🐢"
        status_type = "warning"
    else:
        label_text = str(raw_prediction)
        status_type = "info"

    # Display Result Card
    st.subheader("🎯 Prediction Result")
    
    if status_type == "success":
        st.success(f"### Predicted Movement: **{label_text}**")
    elif status_type == "error":
        st.error(f"### Predicted Movement: **{label_text}**")
    elif status_type == "warning":
        st.warning(f"### Predicted Movement: **{label_text}**")
    else:
        st.info(f"### Predicted Movement: **{label_text}**")

    # 5. Dynamic Probability Display matching model.classes_
    st.write("#### Confidence Breakdown:")
    
    # Map probability array to class names dynamically
    class_prob_map = dict(zip([str(c).lower() for c in model.classes_], probabilities))
    
    fast_prob = class_prob_map.get('fast', class_prob_map.get('0', 0.0)) * 100
    ignored_prob = class_prob_map.get('ignored', class_prob_map.get('1', 0.0)) * 100
    slow_prob = class_prob_map.get('slow', class_prob_map.get('2', 0.0)) * 100

    prob_col1, prob_col2, prob_col3 = st.columns(3)
    prob_col1.metric(label="Fast Probability ⚡", value=f"{fast_prob:.1f}%")
    prob_col2.metric(label="Ignored Probability 💤", value=f"{ignored_prob:.1f}%")
    prob_col3.metric(label="Slow Probability 🐢", value=f"{slow_prob:.1f}%")