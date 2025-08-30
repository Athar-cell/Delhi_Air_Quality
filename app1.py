import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Delhi Air Quality Dashboard 🌫️")

# Load dataset
df = pd.read_csv("dataset.csv")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Filter by year/month
Year = st.selectbox("Select Year", df['Year'].unique())
filtered_df = df[df['Year'] == Year]

st.write(f"Air Quality Data for {Year}")
st.line_chart(filtered_df['PM2.5'])

# Custom Matplotlib plot
fig, ax = plt.subplots()
filtered_df['PM2.5'].plot(kind='hist', bins=30, ax=ax)
st.pyplot(fig)
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("air_quality_model.pkl")

st.title("🌫️ Delhi Air Quality Prediction App")

st.write("Enter air quality parameters below to predict the AQI category:")

# Example inputs
pm25 = st.number_input("PM2.5", min_value=0.0, step=0.1)
pm10 = st.number_input("PM10", min_value=0.0, step=0.1)
no = st.number_input("NO", min_value=0.0, step=0.1)
no2 = st.number_input("NO2", min_value=0.0, step=0.1)
nh3 = st.number_input("NH3", min_value=0.0, step=0.1)
co = st.number_input("CO", min_value=0.0, step=0.1)
so2 = st.number_input("SO2", min_value=0.0, step=0.1)
o3 = st.number_input("O3", min_value=0.0, step=0.1)

if st.button("Predict AQI Bucket"):
    features = np.array([[pm25, pm10, no, no2, nh3, co, so2, o3]])
    prediction = model.predict(features)
    st.success(f"Predicted Air Quality Category: {prediction[0]}")




