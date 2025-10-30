import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="⚡ EV Range Forecaster", page_icon="🚗", layout="wide")

# --- Title and intro ---
st.title("⚡ Electric Vehicle Range Forecaster")
st.markdown("""
Welcome to the **EV Range Forecaster App** 🚗 —  
A Streamlit-powered ML application that predicts the driving range of electric vehicles based on parameters such as battery capacity, temperature, and speed.
""")

# --- Load datasets ---
@st.cache_data
def load_data():
    try:
        df_spec = pd.read_csv("electric_vehicles_spec_2025.csv.csv")
        df_clean = pd.read_csv("cleaned_ev_data.csv")
        df_powerbi = pd.read_csv("EV_Range_Prediction_PowerBI.csv")
        return df_spec, df_clean, df_powerbi
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None, None, None

df_spec, df_clean, df_powerbi = load_data()

if df_spec is not None:
    st.success("✅ Datasets loaded successfully!")
    st.subheader("EV Specifications Preview")
    st.dataframe(df_spec.head())
else:
    st.error("❌ Could not load datasets. Please verify filenames and try again.")

# --- Optional: Dummy ML prediction section ---
st.subheader("🔮 Predict EV Range (Demo)")
battery_capacity = st.number_input("Battery Capacity (kWh)", min_value=10, max_value=150, value=60)
speed = st.slider("Average Speed (km/h)", 20, 140, 60)
temperature = st.slider("Ambient Temperature (°C)", -10, 45, 25)

predicted_range = (battery_capacity * 6) - (speed * 0.5) - (abs(25 - temperature) * 2)
st.metric(label="Estimated Range (km)", value=round(predicted_range, 2))

import streamlit as st
from openai import OpenAI

# Title for Chatbot section
st.subheader("🔋 EV Range Chatbot Assistant")

# Create a text box for user input
user_input = st.text_input("Ask me anything about Electric Vehicles, Range Forecasting, or Battery Efficiency:")

# Only respond when user types something
if user_input:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # make sure you set this in Streamlit secrets
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # you can change to 'gpt-4' if available
            messages=[
                {"role": "system", "content": "You are an expert in electric vehicles and range forecasting."},
                {"role": "user", "content": user_input}
            ]
        )
        st.write("🤖", response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error: {e}")


# --- Footer ---
st.markdown("---")
st.caption("Built by [Sai Charan Naidu P](https://github.com/saicnaidu27) 🚀 | Powered by Streamlit & OpenAI")
