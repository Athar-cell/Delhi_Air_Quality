# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\athar\Downloads\dataset (1).csv")
    df['AQI_Category'] = df['AQI'].apply(lambda aqi: 'Good' if aqi<=50 else 'Satisfactory' if aqi<=100 else 'Moderate' if aqi<=200 else 'Poor' if aqi<=300 else 'Very Poor' if aqi<=400 else 'Severe')
    return df

delhi_dataset = load_data()

# --- Sidebar for user input ---
st.sidebar.header("Enter Delhi Data")
date = st.sidebar.number_input("Date (1-31)", min_value=1, max_value=31, value=1)
month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12, value=1)
year = st.sidebar.number_input("Year (2021-2024)", min_value=2021, max_value=2024, value=2021)
holidays_count = st.sidebar.number_input("Holidays Count (0-1)", min_value=0, max_value=1, value=0)
day = st.sidebar.number_input("Day of Week (1=Mon, 7=Sun)", min_value=1, max_value=7, value=5)
pm25 = st.sidebar.number_input("PM2.5 Level", value=50.0)
pm10 = st.sidebar.number_input("PM10 Level", value=100.0)
no2 = st.sidebar.number_input("NO2 Level", value=40.0)
so2 = st.sidebar.number_input("SO2 Level", value=10.0)
co = st.sidebar.number_input("CO Level", value=1.0)
ozone = st.sidebar.number_input("Ozone Level", value=30.0)

input_data = pd.DataFrame([[month, year, holidays_count, day, pm25, pm10, no2, so2, co, ozone]],
                          columns=['Month','Year','Holidays_Count','Days','PM2.5','PM10','NO2','SO2','CO','Ozone'])

# --- Train model ---
X = delhi_dataset.drop(columns=['Date','AQI','AQI_Category'])
y = delhi_dataset['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict AQI ---
if st.sidebar.button("Predict AQI"):
    predicted_aqi = model.predict(input_data)[0]
    st.subheader(f"Predicted AQI: {predicted_aqi:.2f}")
    if predicted_aqi <=50:
        st.success("Air Quality: Good")
    elif predicted_aqi <=100:
        st.info("Air Quality: Satisfactory")
    elif predicted_aqi <=200:
        st.warning("Air Quality: Moderate")
    elif predicted_aqi <=300:
        st.error("Air Quality: Poor")
    elif predicted_aqi <=400:
        st.error("Air Quality: Very Poor")
    else:
        st.error("Air Quality: Severe")

# --- Visualization Section ---
st.header("Visualizations")
if st.checkbox("Show AQI Distribution by Month"):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=delhi_dataset, x='Month', y='AQI', palette='coolwarm')
    st.pyplot(plt)

if st.checkbox("Show AQI vs Holidays Count"):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Holidays_Count', y='AQI', data=delhi_dataset, palette='Set2')
    st.pyplot(plt)

if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10,8))
    correlation = delhi_dataset[['PM2.5','PM10','NO2','SO2','CO','Ozone','AQI']].corr()
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', fmt='.2f')
    st.pyplot(plt)

if st.checkbox("Show Monthly Average Pollutant Levels"):
    monthly_avg = delhi_dataset.groupby('Month')[['PM2.5','PM10','NO2','SO2','CO','Ozone']].mean()
    st.bar_chart(monthly_avg)

