import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from openai import OpenAI
import os

st.set_page_config(page_title="EV Range Assistant", layout="wide")
st.title("⚡ Electric Vehicle Range Prediction & AI Assistant")

# Load dataset
df = pd.read_excel(r"C:\Users\saich\Documents\Shell_Edunet\electric_vehicles_spec_2025.xlsx")
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Feature selection
features = ['battery_capacity_kWh', 'top_speed_kmh', 'torque_nm', 'efficiency_wh_per_km']
target = 'range_km'
df = df[features + [target]].dropna()

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.success(f"✅ Model Trained | R²: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Input form for prediction
st.header("🔋 Predict EV Range")
col1, col2, col3, col4 = st.columns(4)
with col1:
    battery = st.number_input("Battery (kWh)", 20, 150, 60)
with col2:
    speed = st.number_input("Top Speed (km/h)", 80, 300, 150)
with col3:
    torque = st.number_input("Torque (Nm)", 100, 1000, 300)
with col4:
    efficiency = st.number_input("Efficiency (Wh/km)", 100, 300, 180)

if st.button("🚗 Predict Range"):
    pred = model.predict([[battery, speed, torque, efficiency]])[0]
    st.success(f"🔹 Estimated Range: {pred:.2f} km")

# ---------------- Chatbot Section ----------------
st.header("🤖 EV Chat Assistant")

user_question = st.text_input("Ask anything about Electric Vehicles:")
if user_question:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Add it in C:\\Users\\saich\\.streamlit\\secrets.toml")
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with st.spinner("AI is thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an EV expert assistant."},
                    {"role": "user", "content": user_question}
                ]
            )
        st.info(response.choices[0].message.content)
