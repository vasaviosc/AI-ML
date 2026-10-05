from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(MODELS_DIR / "house_price_model.pkl")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the details of the house to predict its price.")

area = st.number_input(
    "Area (sq ft)",
    min_value=0,
    value=5000
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=0,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=0,
    value=2,
    step=1
)

stories = st.number_input(
    "Number of Stories",
    min_value=0,
    value=2,
    step=1
)

mainroad = st.selectbox(
    "Main Road",
    ["yes", "no"]
)

guestroom = st.selectbox(
    "Guest Room",
    ["yes", "no"]
)

basement = st.selectbox(
    "Basement",
    ["yes", "no"]
)

hotwaterheating = st.selectbox(
    "Hot Water Heating",
    ["yes", "no"]
)

airconditioning = st.selectbox(
    "Air Conditioning",
    ["yes", "no"]
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    value=2,
    step=1
)

prefarea = st.selectbox(
    "Preferred Area",
    ["yes", "no"]
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

input_data = pd.get_dummies(
    input_data,
    drop_first=True
)

input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

if st.button("Predict House Price"):
    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: ₹{prediction[0]:,.2f}"
    )