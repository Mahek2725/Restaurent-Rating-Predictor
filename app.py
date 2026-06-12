import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('restaurant_rating_model(5).pkl')
encoders = joblib.load('encoders(7).pkl')

st.title("🍽️ Restaurant Rating Predictor")

st.write("Enter restaurant details to predict the aggregate rating.")

# Numeric Inputs
country_code = st.number_input("Country Code", value=1)
longitude = st.number_input("Longitude", value=0.0)
latitude = st.number_input("Latitude", value=0.0)
avg_cost = st.number_input("Average Cost for Two", value=500)
price_range = st.number_input("Price Range", min_value=1, max_value=4, value=2)
votes = st.number_input("Votes", value=100)

# Categorical Inputs
city = st.selectbox("City", encoders['City'].classes_)
locality = st.selectbox("Locality", encoders['Locality'].classes_)
cuisines = st.selectbox("Cuisines", encoders['Cuisines'].classes_)
currency = st.selectbox("Currency", encoders['Currency'].classes_)

table_booking = st.selectbox(
    "Has Table Booking",
    encoders['Has Table booking'].classes_
)

online_delivery = st.selectbox(
    "Has Online Delivery",
    encoders['Has Online delivery'].classes_
)

delivering_now = st.selectbox(
    "Is Delivering Now",
    encoders['Is delivering now'].classes_
)

switch_menu = st.selectbox(
    "Switch To Order Menu",
    encoders['Switch to order menu'].classes_
)

rating_color = st.selectbox(
    "Rating Color",
    encoders['Rating color'].classes_
)

rating_text = st.selectbox(
    "Rating Text",
    encoders['Rating text'].classes_
)

if st.button("Predict Rating"):

    input_data = pd.DataFrame([{
        'Country Code': country_code,
        'City': encoders['City'].transform([city])[0],
        'Locality': encoders['Locality'].transform([locality])[0],
        'Longitude': longitude,
        'Latitude': latitude,
        'Cuisines': encoders['Cuisines'].transform([cuisines])[0],
        'Average Cost for two': avg_cost,
        'Currency': encoders['Currency'].transform([currency])[0],
        'Has Table booking': encoders['Has Table booking'].transform([table_booking])[0],
        'Has Online delivery': encoders['Has Online delivery'].transform([online_delivery])[0],
        'Is delivering now': encoders['Is delivering now'].transform([delivering_now])[0],
        'Switch to order menu': encoders['Switch to order menu'].transform([switch_menu])[0],
        'Price range': price_range,
        'Rating color': encoders['Rating color'].transform([rating_color])[0],
        'Rating text': encoders['Rating text'].transform([rating_text])[0],
        'Votes': votes
    }])

    prediction = model.predict(input_data)

    st.success(f"Predicted Aggregate Rating: {prediction[0]:.2f}")
