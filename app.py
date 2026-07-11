import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. LOAD THE MODEL & COLUMNS
# ---------------------------------------------------------
@st.cache_resource # This keeps the model in memory so it doesn't reload every click
def load_assets():
    model = joblib.load('xgboost_f1_model.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, cols

xgb_model, expected_columns = load_assets()

# ---------------------------------------------------------
# 2. DYNAMICALLY EXTRACT UI OPTIONS
# ---------------------------------------------------------
# Find all columns that start with 'Track_', 'Driver_', or 'Compound_' 
# and strip the prefix to get the clean names for the dropdowns
tracks = [col.replace('Track_', '') for col in expected_columns if col.startswith('Track_')]
drivers = [col.replace('Driver_', '') for col in expected_columns if col.startswith('Driver_')]
compounds = [col.replace('Compound_', '') for col in expected_columns if col.startswith('Compound_')]

tracks.sort()
drivers.sort()

# ---------------------------------------------------------
# 3. BUILD THE USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="F1 Strategy Predictor", layout="centered")

st.title("F1 Lap Time Predictor")

col1, col2 = st.columns(2)

with col1:
    selected_track = st.selectbox("Race Track", tracks)
    selected_driver = st.selectbox("Driver", drivers)
    selected_compound = st.radio("Tyre Compound", compounds)

with col2:
    tyre_age = st.slider("Tyre Age (Laps)", min_value=1, max_value=50, value=15)
    track_temp = st.slider("Track Temp (°C)", min_value=15.0, max_value=60.0, value=35.0, step=1.0)
    air_temp = st.slider("Air Temp (°C)", min_value=10.0, max_value=45.0, value=25.0, step=1.0)
    # Create a binary choice for the user
    track_condition = st.radio("Track Condition", ["Dry", "Wet"])
    # Translate their choice into the 0 or 1 the model needs
    rainfall = 1.0 if track_condition == "Wet" else 0.0    

# ---------------------------------------------------------
# 4. PREDICTION LOGIC
# ---------------------------------------------------------
if st.button("Predict Lap Time", type="primary"):
    
    # Create a base dictionary with the numerical inputs
    input_dict = {
        'TyreAge': tyre_age,
        'TrackTemp': track_temp,
        'AirTemp': air_temp,
        'Rainfall': rainfall
    }
    
    # Activate the specific one-hot columns the user selected
    input_dict[f'Track_{selected_track}'] = 1
    input_dict[f'Driver_{selected_driver}'] = 1
    input_dict[f'Compound_{selected_compound}'] = 1
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # CRITICAL: Reindex aligns our 7 inputs to the 50+ columns the model expects,
    # automatically filling all the unselected tracks and drivers with 0.
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    # Run Prediction
    prediction = xgb_model.predict(input_df)[0]
    
    # Display the result
    st.divider()
    st.subheader("⏱️ Predicted Lap Time:")
    st.metric(label="", value=f"{prediction:.3f} seconds")
    
    # Contextual feedback for the user
    if tyre_age > 20 and selected_compound == "Soft":
        st.warning("⚠️ Warning: Soft compound at this age usually experiences severe degradation.")