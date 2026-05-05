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
/* Optional: Keep Streamlit's default header and footer visible */

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

# List the specific features you want to show in the UI for the user to input
# (Right now it's just taking the first 6 features from the list, but you can change this to a list of exact names like ["LotArea", "OverallQual", ...])
user_input_features = ['GrLivArea', 'YrBltAndRemod', 'TotalBsmtSF', 'GarageArea', 'FireplaceQu_NA']

# Hardcoded default values for variables NOT shown in the UI
# If a feature is not in this dictionary, it will just default to 0.0
hardcoded_defaults = {
    'LotFrontage': 69.8637,
    'LotArea': 10516.8281,
    'Street': 0.0000,
    'Utilities': 0.0000,
    'Condition2': 0.0000,
    'YearBuilt': 1971.2678,
    'YearRemodAdd': 1984.8658,
    'RoofMatl': 0.0000,
    'MasVnrArea': 103.1171,
    'BsmtFinSF1': 443.6397,
    'BsmtFinSF2': 46.5493,
    'BsmtUnfSF': 567.2404,
    'Heating': 0.0000,
    'CentralAir': 0.0000,
    '1stFlrSF': 1162.6267,
    '2ndFlrSF': 346.9925,
    'LowQualFinSF': 5.8445,
    'GarageYrBlt': 1978.5890,
    'WoodDeckSF': 94.2445,
    'OpenPorchSF': 46.6603,
    'EnclosedPorch': 21.9541,
    '3SsnPorch': 3.4096,
    'ScreenPorch': 15.0610,
    'PoolArea': 2.7589,
    'PoolQC': 0.0000,
    'MiscVal': 43.4890,
    'Totallot': 0.0000,
    'MSSubClass_160': 0.0432,
    'MSSubClass_190': 0.0205,
    'MSSubClass_20': 0.3671,
    'MSSubClass_30': 0.0473,
    'MSSubClass_50': 0.0986,
    'MSSubClass_60': 0.2048,
    'MSSubClass_70': 0.0411,
    'MSSubClass_80': 0.0397,
    'MSSubClass_85': 0.0137,
    'MSSubClass_90': 0.0356,
    'MSSubClass_Rare': 0.0000,
    'MSZoning_RL': 0.7884,
    'MSZoning_RM': 0.1493,
    'MSZoning_Rare': 0.0000,
    'Alley_NA': 0.0000,
    'Alley_Pave': 0.0281,
    'LotShape_IR2': 0.0281,
    'LotShape_Rare': 0.0000,
    'LotShape_Reg': 0.6336,
    'LandContour_HLS': 0.0342,
    'LandContour_Low': 0.0247,
    'LandContour_Lvl': 0.8979,
    'LotConfig_CulDSac': 0.0644,
    'LotConfig_FR2': 0.0322,
    'LotConfig_Inside': 0.7205,
    'LotConfig_Rare': 0.0000,
    'LandSlope_Mod': 0.0445,
    'LandSlope_Rare': 0.0000,
    'Neighborhood_BrkSide': 0.0397,
    'Neighborhood_ClearCr': 0.0192,
    'Neighborhood_CollgCr': 0.1027,
    'Neighborhood_Crawfor': 0.0349,
    'Neighborhood_Edwards': 0.0685,
    'Neighborhood_Gilbert': 0.0541,
    'Neighborhood_IDOTRR': 0.0253,
    'Neighborhood_MeadowV': 0.0116,
    'Neighborhood_Mitchel': 0.0336,
    'Neighborhood_NAmes': 0.1541,
    'Neighborhood_NWAmes': 0.0500,
    'Neighborhood_NoRidge': 0.0281,
    'Neighborhood_NridgHt': 0.0527,
    'Neighborhood_OldTown': 0.0774,
    'Neighborhood_Rare': 0.0000,
    'Neighborhood_SWISU': 0.0171,
    'Neighborhood_Sawyer': 0.0507,
    'Neighborhood_SawyerW': 0.0404,
    'Neighborhood_Somerst': 0.0589,
    'Neighborhood_StoneBr': 0.0171,
    'Neighborhood_Timber': 0.0260,
    'Condition1_Feedr': 0.0555,
    'Condition1_Norm': 0.8630,
    'Condition1_PosN': 0.0130,
    'Condition1_RRAn': 0.0178,
    'Condition1_Rare': 0.0000,
    'BldgType_2fmCon': 0.0212,
    'BldgType_Duplex': 0.0356,
    'BldgType_Twnhs': 0.0295,
    'BldgType_TwnhsE': 0.0781,
    'HouseStyle_1Story': 0.4973,
    'HouseStyle_2Story': 0.3048,
    'HouseStyle_Rare': 0.0000,
    'HouseStyle_SFoyer': 0.0253,
    'HouseStyle_SLvl': 0.0445,
    'OverallQual_2.0': 0.0021,
    'OverallQual_3.0': 0.0137,
    'OverallQual_4.0': 0.0795,
    'OverallQual_5.0': 0.2719,
    'OverallQual_6.0': 0.2562,
    'OverallQual_7.0': 0.2185,
    'OverallQual_8.0': 0.1151,
    'OverallQual_9.0': 0.0295,
    'OverallQual_10.0': 0.0123,
    'OverallCond_4': 0.0390,
    'OverallCond_5': 0.5623,
    'OverallCond_6': 0.1726,
    'OverallCond_7': 0.1404,
    'OverallCond_8': 0.0493,
    'OverallCond_9': 0.0151,
    'OverallCond_Rare': 0.0000,
    'RoofStyle_Hip': 0.1959,
    'RoofStyle_Rare': 0.0000,
    'Exterior1st_BrkFace': 0.0342,
    'Exterior1st_CemntBd': 0.0418,
    'Exterior1st_HdBoard': 0.1521,
    'Exterior1st_MetalSd': 0.1507,
    'Exterior1st_Plywood': 0.0740,
    'Exterior1st_Rare': 0.0000,
    'Exterior1st_Stucco': 0.0171,
    'Exterior1st_VinylSd': 0.3527,
    'Exterior1st_Wd Sdng': 0.1411,
    'Exterior1st_WdShing': 0.0178,
    'Exterior2nd_BrkFace': 0.0171,
    'Exterior2nd_CmentBd': 0.0411,
    'Exterior2nd_HdBoard': 0.1418,
    'Exterior2nd_MetalSd': 0.1466,
    'Exterior2nd_Plywood': 0.0973,
    'Exterior2nd_Rare': 0.0000,
    'Exterior2nd_Stucco': 0.0178,
    'Exterior2nd_VinylSd': 0.3452,
    'Exterior2nd_Wd Sdng': 0.1349,
    'Exterior2nd_Wd Shng': 0.0260,
    'MasVnrType_None': 0.0000,
    'MasVnrType_Rare': 0.0000,
    'MasVnrType_Stone': 0.0877,
    'ExterQual_Fa': 0.0096,
    'ExterQual_Gd': 0.3342,
    'ExterQual_TA': 0.6205,
    'ExterCond_Gd': 0.1000,
    'ExterCond_Rare': 0.0000,
    'ExterCond_TA': 0.8781,
    'Foundation_CBlock': 0.4342,
    'Foundation_PConc': 0.4432,
    'Foundation_Rare': 0.0000,
    'Foundation_Slab': 0.0164,
    'BsmtQual_Fa': 0.0240,
    'BsmtQual_Gd': 0.4233,
    'BsmtQual_NA': 0.0000,
    'BsmtQual_TA': 0.4699,
    'BsmtCond_Gd': 0.0445,
    'BsmtCond_NA': 0.0000,
    'BsmtCond_Rare': 0.0000,
    'BsmtCond_TA': 0.9233,
    'BsmtExposure_Gd': 0.0918,
    'BsmtExposure_Mn': 0.0781,
    'BsmtExposure_NA': 0.0000,
    'BsmtExposure_No': 0.6788,
    'BsmtFinType1_BLQ': 0.1014,
    'BsmtFinType1_GLQ': 0.2863,
    'BsmtFinType1_LwQ': 0.0507,
    'BsmtFinType1_NA': 0.0000,
    'BsmtFinType1_Rec': 0.0911,
    'BsmtFinType1_Unf': 0.3199,
    'BsmtFinType2_BLQ': 0.0226,
    'BsmtFinType2_GLQ': 0.0096,
    'BsmtFinType2_LwQ': 0.0315,
    'BsmtFinType2_NA': 0.0000,
    'BsmtFinType2_Rec': 0.0370,
    'BsmtFinType2_Unf': 0.8863,
    'HeatingQC_Fa': 0.0336,
    'HeatingQC_Gd': 0.1651,
    'HeatingQC_Rare': 0.0000,
    'HeatingQC_TA': 0.2932,
    'Electrical_FuseF': 0.0185,
    'Electrical_Rare': 0.0000,
    'Electrical_SBrkr': 0.9144,
    'BsmtFullBath_1.0': 0.4027,
    'BsmtFullBath_2.0': 0.0103,
    'BsmtFullBath_3.0': 0.0007,
    'BsmtHalfBath_1.0': 0.0548,
    'BsmtHalfBath_2.0': 0.0014,
    'FullBath_1': 0.4452,
    'FullBath_2': 0.5260,
    'FullBath_3': 0.0226,
    'FullBath_4': 0.0000,
    'HalfBath_1': 0.3664,
    'HalfBath_2': 0.0082,
    'BedroomAbvGr_1': 0.0342,
    'BedroomAbvGr_2': 0.2452,
    'BedroomAbvGr_3': 0.5507,
    'BedroomAbvGr_4': 0.1459,
    'BedroomAbvGr_5': 0.0144,
    'BedroomAbvGr_6': 0.0048,
    'BedroomAbvGr_8': 0.0007,
    'KitchenAbvGr_1': 0.9534,
    'KitchenAbvGr_2': 0.0445,
    'KitchenAbvGr_3': 0.0014,
    'KitchenQual_Fa': 0.0267,
    'KitchenQual_Gd': 0.4014,
    'KitchenQual_TA': 0.5034,
    'TotRmsAbvGrd_3.0': 0.0116,
    'TotRmsAbvGrd_4.0': 0.0664,
    'TotRmsAbvGrd_5.0': 0.1884,
    'TotRmsAbvGrd_6.0': 0.2753,
    'TotRmsAbvGrd_7.0': 0.2253,
    'TotRmsAbvGrd_8.0': 0.1281,
    'TotRmsAbvGrd_9.0': 0.0514,
    'TotRmsAbvGrd_10.0': 0.0322,
    'TotRmsAbvGrd_11.0': 0.0123,
    'TotRmsAbvGrd_12.0': 0.0075,
    'TotRmsAbvGrd_12.5': 0.0000,
    'Functional_Min2': 0.0233,
    'Functional_Mod': 0.0103,
    'Functional_Rare': 0.0000,
    'Functional_Typ': 0.9315,
    'Fireplaces_1': 0.4452,
    'Fireplaces_2': 0.0788,
    'Fireplaces_3': 0.0034,
    'Fireplaces_4': 0.0000,
    'FireplaceQu_Fa': 0.0226,
    'FireplaceQu_Gd': 0.7329,
    'FireplaceQu_Po': 0.0137,
    'FireplaceQu_TA': 0.2144,
    'GarageType_Basment': 0.0130,
    'GarageType_BuiltIn': 0.0603,
    'GarageType_Detchd': 0.2651,
    'GarageType_NA': 0.0000,
    'GarageType_Rare': 0.0000,
    'GarageFinish_NA': 0.0000,
    'GarageFinish_RFn': 0.2890,
    'GarageFinish_Unf': 0.4699,
    'GarageCars_1.0': 0.2527,
    'GarageCars_2.0': 0.5644,
    'GarageCars_3.0': 0.1240,
    'GarageCars_4.0': 0.0034,
    'GarageCars_5.0': 0.0000,
    'GarageQual_NA': 0.0000,
    'GarageQual_Rare': 0.0000,
    'GarageQual_TA': 0.9534,
    'GarageCond_NA': 0.0000,
    'GarageCond_Rare': 0.0000,
    'GarageCond_TA': 0.9637,
    'PavedDrive_P': 0.0205,
    'PavedDrive_Y': 0.9178,
    'Fence_GdWo': 0.0370,
    'Fence_MnPrv': 0.9151,
    'Fence_NA': 0.0000,
    'Fence_Rare': 0.0000,
    'MiscFeature_Rare': 0.0000,
    'MiscFeature_Shed': 0.9966,
    'MoSold_10': 0.0610,
    'MoSold_11': 0.0541,
    'MoSold_12': 0.0404,
    'MoSold_2': 0.0356,
    'MoSold_3': 0.0726,
    'MoSold_4': 0.0966,
    'MoSold_5': 0.1397,
    'MoSold_6': 0.1733,
    'MoSold_7': 0.1603,
    'MoSold_8': 0.0836,
    'MoSold_9': 0.0432,
    'YrSold_2007': 0.2253,
    'YrSold_2008': 0.2082,
    'YrSold_2009': 0.2315,
    'YrSold_2010': 0.1199,
    'SaleType_New': 0.0836,
    'SaleType_Rare': 0.0000,
    'SaleType_WD': 0.8678,
    'SaleCondition_Family': 0.0137,
    'SaleCondition_Normal': 0.8205,
    'SaleCondition_Partial': 0.0856,
    'SaleCondition_Rare': 0.0000,
    'Total_Bathrooms_2.0': 0.0000,
    'Total_Bathrooms_3.0': 0.0000,
    'Total_Bathrooms_4.0': 0.0000,
    'Total_Bathrooms_5.0': 0.0000,
    'Total_Bathrooms_6.0': 0.0000,
    'Total_Bathrooms_8.0': 0.0000,
    'Qual_Area_Interact': 0.0000,
}

# Dictionary to hold the final inputs that go to the model
input_data = {}

# Create a 3-column layout to make it look nicer
col1, col2, col3 = st.columns(3)

col_index = 0
for feature in features:
    if feature in user_input_features:
        # Display an input field in the UI for this feature
        if col_index % 3 == 0:
            with col1:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
        elif col_index % 3 == 1:
            with col2:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
        else:
            with col3:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
        col_index += 1
    else:
        # Do not show in UI, just assign the hardcoded value (or 0.0 if not specified)
        input_data[feature] = hardcoded_defaults.get(feature, 0.0)

st.markdown("---")

# 3. Predict Button
if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        # Convert user inputs into a single-row Pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure the columns are in the EXACT same order as they were during training
        input_df = input_df[features]
        
        # The scaler was only fit on numeric features, so we only apply it to those features!
        numeric_cols = scaler.feature_names_in_
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Make the prediction using the complete dataframe
        prediction = model.predict(input_df)
        
        # Note: In your EDA, you looked at np.log(SalePrice). 
        # IF you trained your model on the log of the price, you must reverse it here using np.expm1()
        # prediction_final = np.expm1(prediction[0]) 
        # If you trained on the normal price, just use prediction[0]
        prediction_final = prediction[0] 
        
        st.success(f"### Estimated Price: $M{prediction_final:,.2f}")
        
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