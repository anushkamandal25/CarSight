import streamlit as st
import pandas as pd
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fefae0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


pipeline = joblib.load('car_price_predictor.joblib')


data = pd.read_excel('result1.xlsx')


st.title('Car Price Prediction')


col1, col2 = st.columns(2)

with col1:
    car_model = st.selectbox("Select Model", sorted(data['Model'].dropna().unique()))
    year = st.number_input("Year of Registration", min_value=1990, max_value=2025, value=2015)
    engine_size = st.number_input("Engine Size (cc)", min_value=600, max_value=5000, value=2000)
    seats = st.number_input("Number of Seats", min_value=1, max_value=20, value=5)
    drive = st.selectbox("Drive", sorted(data['Drive'].dropna().unique()))
    transmission = st.selectbox("Transmission", data['Transmission'].dropna().unique())

with col2:
    car_brand = st.text_input("Car Brand", value="Toyota")
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=50000)
    fuel_type = st.selectbox("Fuel Type", data['Fuel'].dropna().unique())
    doors = st.number_input("Number of Doors", min_value=2, max_value=6, value=4)
    ext_color = st.selectbox("Exterior Color", sorted(data['Ext_Color'].dropna().unique()))
    steering = st.selectbox("Steering", sorted(data['Steering'].dropna().unique()))

# Create input DataFrame for prediction
if st.button('Predict Price'):
    vehicle_age = 2025 - year
    input_df = pd.DataFrame([{
        'Model': car_model,
        'Brand': car_brand,
        'Mileage': mileage,
        'Engine_Size': engine_size,
        'Fuel': fuel_type,
        'Transmission': transmission,
        'Vehicle_Age': vehicle_age,
        'Seats': seats,
        'Doors': doors,
        'Drive': drive,
        'Ext_Color': ext_color,
        'Steering': steering
    }])

    # --- PATCH: Ensure input_df has same columns & encoding as training ---
    try:
        pred_price = pipeline.predict(input_df)[0]
        st.success(f"Predicted Price: ${pred_price:,.2f}")
    except (ValueError, KeyError) as e:
        st.error("Prediction failed. Please check input format.")
        st.exception(e)


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        text-align: center;
        color: gray;
        font-size: 14px;
        padding: 10px 0;
    }
    </style>
    <div class="footer">
        Made with ❤️ Anushka Mandal © 2025
    </div>
    """,
    unsafe_allow_html=True
)
