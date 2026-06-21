import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. Synthesize Data & Train Model ---
@st.cache_resource
def get_trained_model():
    np.random.seed(42)
    distance = np.random.uniform(1, 15, 200) 
    rating = np.random.uniform(3.0, 5.0, 200) 
    rush_hour = np.random.choice([0, 1], 200) 

    # Life-like delivery time simulation formula
    time = 10 + (4 * distance) - (2 * rating) + (20 * rush_hour) + np.random.normal(0, 3, 200)

    df = pd.DataFrame({
        'distance': distance, 
        'rating': rating, 
        'rush_hour': rush_hour, 
        'time': time
    })

    # Train our best model (Model 3 with traffic data)
    X = df[['distance', 'rating', 'rush_hour']]
    y = df['time']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Load the model
model = get_trained_model()

# --- 2. Streamlit User Interface ---
st.set_page_config(page_title="Delivery Predictor", page_icon="🛵")

st.title("Delivery Time Predictor 🛵")
st.write("A Machine Learning app to estimate delivery times using Linear Regression.")

# Form for user inputs
with st.form("prediction_form"):
    st.subheader("Enter Delivery Details")
    
    user_distance = st.number_input("Distance (in km)", min_value=1.0, max_value=25.0, value=5.0, step=0.5)
    user_rating = st.slider("Rider Rating (Stars)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    user_rush_hour = st.selectbox("Is it Rush Hour traffic?", ["No", "Yes"])
    
    # Form submit button
    submit_button = st.form_submit_button(label="Calculate Delivery Time")

# --- 3. Run Prediction ---
if submit_button:
    # Convert text selection to binary numeric input (0 or 1)
    rush_hour_numeric = 1 if user_rush_hour == "Yes" else 0
    
    # Make prediction
    features = [[user_distance, user_rating, rush_hour_numeric]]
    predicted_time = model.predict(features)[0]
    
    # Display result clearly
    st.success(f"### Estimated Delivery Time: **{predicted_time:.0f} minutes**")