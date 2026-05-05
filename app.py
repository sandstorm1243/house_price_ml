# import streamlit as st
# import joblib
# import pandas as pd

# # 1. Load your saved assets
# model = joblib.load('house_model.pkl')
# scaler = joblib.load('scaler.pkl')
# features = joblib.load('features.pkl')

# st.title("🏠 House Price Predictor")
# st.write("Enter the property details below to get an estimated price.")

# # 2. Create input fields dynamically based on your features.pkl
# # This loops through every feature your model expects and creates a number box
# st.header("Property Details")
# input_data = {}

# # Using columns to make the layout look nice (2 columns wide)
# col1, col2 = st.columns(2)

# for i, feature in enumerate(features):
#     # Alternate putting inputs in column 1 and column 2
#     with col1 if i % 2 == 0 else col2:
#         input_data[feature] = st.number_input(f"{feature}", value=0.0)

# # 3. Predict Button
# if st.button("Predict Price", type="primary"):
    
#     # Convert the user's inputs into a Pandas DataFrame using the exact feature order
#     input_df = pd.DataFrame([input_data])
    
#     # Scale the input data using your saved scaler
#     scaled_input = scaler.transform(input_df)
    
#     # Make the prediction
#     prediction = model.predict(scaled_input)
    
#     # Display the result formatted as currency
#     st.success(f"Estimated Price: ${prediction[0]:,.2f}")




import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Custom CSS for modern UI, animations, header, and footer
custom_css = """
<style>
/* Hide default header and footer */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.custom-header {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    animation: fadeInDown 1s ease-out;
}
.custom-header h1 {
    margin: 0;
    font-size: 2.8rem;
    font-weight: 800;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    color: white !important;
}

.custom-footer {
    background-color: var(--secondary-background-color);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-top: 4rem;
    animation: fadeInUp 1s ease-out;
    box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
}
.custom-footer p {
    font-weight: bold;
    margin-bottom: 1rem;
    color: var(--text-color);
    font-size: 1.2rem;
}
.contributors {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 1rem;
}
.contributors span {
    display: inline-block;
    padding: 0.8rem 1.5rem;
    background-color: var(--background-color);
    color: var(--text-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 25px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    cursor: default;
}
.contributors span:hover {
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white !important;
    border-color: transparent;
}

/* Hover effects for inputs */
div[data-baseweb="input"] {
    transition: all 0.3s ease;
}
div[data-baseweb="input"]:focus-within {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Custom Header
st.markdown("""
<div class="custom-header">
    <h1>🏠 House Price Prediction ML Model</h1>
</div>
""", unsafe_allow_html=True)

st.write("### Enter the property details below to get an estimated price.")

# 1. Load the saved assets with error handling
@st.cache_resource # This prevents Streamlit from reloading the model every time you click a button
def load_assets():
    try:
        model = joblib.load('house_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model, scaler, features
    except Exception as e:
        st.error(f"Could not load model files. Please ensure house_model.pkl, scaler.pkl, and features.pkl are in the same folder. Error: {e}")
        st.stop()

model, scaler, features = load_assets()

# 2. Create the input form dynamically
st.header("Property Details")

# Dictionary to hold user inputs, initializing all required features to 0.0
input_data = {feature: 0.0 for feature in features}

# Create a 3-column layout to make it look nicer
col1, col2, col3 = st.columns(3)

# Loop through ONLY the first 6 features the model requires and create a number input
for i, feature in enumerate(features[:6]):
    if i % 3 == 0:
        with col1:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    elif i % 3 == 1:
        with col2:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    else:
        with col3:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)

st.markdown("---")

# 3. Predict Button
if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        # Convert user inputs into a single-row Pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure the columns are in the EXACT same order as they were during training
        input_df = input_df[features]
        
        # Apply your saved scaler
        scaled_input = scaler.transform(input_df)
        
        # Make the prediction
        prediction = model.predict(scaled_input)
        
        # Note: In your EDA, you looked at np.log(SalePrice). 
        # IF you trained your model on the log of the price, you must reverse it here using np.expm1()
        # prediction_final = np.expm1(prediction[0]) 
        # If you trained on the normal price, just use prediction[0]
        prediction_final = prediction[0] 
        
        st.success(f"### Estimated Price: ${prediction_final:,.2f}")
        
    except Exception as e:
        st.error("An error occurred during prediction.")
        st.write(f"**Error Details:** {e}")
        st.info("Check your inputs. Did you include text features in your model without encoding them first? Are there any missing values?")

# Custom Footer
st.markdown("""
<div class="custom-footer">
    <p>👨‍💻 Contributors</p>
    <div class="contributors">
        <span>Aditya Srivastav<br><small>Roll: 236301024</small></span>
        <span>Aayush Rawat<br><small>Roll: 236301008</small></span>
        <span>Shanti Bhushan<br><small>Roll: 236301191</small></span>
    </div>
</div>
""", unsafe_allow_html=True)