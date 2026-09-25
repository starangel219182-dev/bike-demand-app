import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Bike Share Dispatcher", page_icon="🚲", layout="centered")

st.title("🚲 Urban Bike Demand Prediction")
st.markdown("Live client querying the **FastAPI `@champion`** inference endpoint.")

# Health check badge
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    if health.get("model_loaded"):
        st.success("✅ Prediction API is Online (Model: `@champion`)")
    else:
        st.warning("⚠️ Prediction API is running, but model is not loaded.")
except Exception:
    st.error("❌ Prediction API is Offline. Run `uvicorn service.app:app` first.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    hr = st.slider("Hour of Day (0-23)", 0, 23, 17)
    temp_c = st.slider("Temperature (°C)", 0.0, 40.0, 24.0)
    hum_pct = st.slider("Humidity (%)", 0, 100, 50)
    wind_kmh = st.slider("Windspeed (km/h)", 0.0, 50.0, 12.0)
    season_val = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])

with col2:
    workingday = st.radio("Calendar Type", [1, 0], format_func=lambda x: "Working Day" if x == 1 else "Weekend / Holiday")
    weather = st.selectbox("Weather Condition", [1, 2, 3, 4], format_func=lambda x: {1: "Clear / Few Clouds", 2: "Mist / Cloudy", 3: "Light Rain / Snow", 4: "Heavy Rain"}[x])
    weekday = st.slider("Day of Week (0=Sun, 6=Sat)", 0, 6, 2)
    month = st.slider("Month (1-12)", 1, 12, 6)
    year = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")

# Scale parameters to normalized values as in the dataset (temp/41, atemp/50, hum/100, windspeed/67)
payload = {
    "season": season_val,
    "yr": year,
    "mnth": month,
    "hr": hr,
    "holiday": 0,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weather,
    "temp": round(temp_c / 41.0, 4),
    "atemp": round(temp_c / 50.0, 4),
    "hum": round(hum_pct / 100.0, 4),
    "windspeed": round(wind_kmh / 67.0, 4)
}

if st.button("🚀 Predict Hourly Demand", use_container_width=True):
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if res.status_code == 200:
            count = res.json()["predicted_demand"]
            st.metric("Predicted Rental Bikes Needed", f"{count} bikes")
            if count > 300:
                st.warning("⚠️ High Volume Period: Consider staging redistribution trucks.")
            else:
                st.info("ℹ️ Standard operating volume.")
        else:
            st.error(f"Prediction failed: {res.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")