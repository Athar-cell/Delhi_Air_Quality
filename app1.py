# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="Delhi Air Quality Dashboard", layout="wide")
st.title("Delhi Air Quality Analysis & Prediction")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your Delhi AQI CSV file", type="csv")

if uploaded_file is not None:
    delhi_dataset = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(delhi_dataset.head())

    # --- Data summary ---
    st.subheader("Dataset Info")
    st.write(delhi_dataset.describe())

    st.subheader("Missing Values Check")
    st.write(delhi_dataset.isnull().sum())

    # --- Visualizations ---
    st.subheader("Daily Trend of Pollutants")
    plt.figure(figsize=(14,6))
    for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']:
        plt.plot(delhi_dataset['Date'], delhi_dataset[col], label=col, alpha=0.7)
    plt.title('Daily Trend of Pollutants')
    plt.xlabel('Date')
    plt.ylabel('Pollutant Levels')
    plt.legend()
    st.pyplot(plt)

    st.subheader("AQI Distribution by Month")
    plt.figure(figsize=(10,6))
    sns.boxplot(data=delhi_dataset, x='Month', y='AQI', palette='coolwarm')
    plt.xlabel('Month')
    plt.ylabel('AQI')
    st.pyplot(plt)

    st.subheader("AQI vs Number of Holidays")
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Holidays_Count', y='AQI', data=delhi_dataset, palette='Set2')
    plt.xlabel('Holidays Count')
    plt.ylabel('AQI')
    st.pyplot(plt)

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10,8))
    correlation = delhi_dataset[['PM2.5','PM10','NO2','SO2','CO','Ozone','AQI']].corr()
    sns.heatmap(correlation, annot=True, cmap='RdBu_r', fmt=".2f")
    st.pyplot(plt)

    # --- Average monthly pollutant levels ---
    st.subheader("Average Monthly Pollutant Levels")
    monthly_avg = delhi_dataset.groupby('Month')[['PM2.5','PM10','NO2','SO2','CO','Ozone']].mean()
    st.bar_chart(monthly_avg)

    # --- AQI Categories ---
    st.subheader("AQI Category Distribution")
    def categorize_aqi(aqi):
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Satisfactory'
        elif aqi <= 200: return 'Moderate'
        elif aqi <= 300: return 'Poor'
        elif aqi <= 400: return 'Very Poor'
        else: return 'Severe'

    delhi_dataset['AQI_Category'] = delhi_dataset['AQI'].apply(categorize_aqi)
    aqi_counts = delhi_dataset['AQI_Category'].value_counts()
    plt.figure(figsize=(7,7))
    plt.pie(aqi_counts, labels=aqi_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
    st.pyplot(plt)

    # --- ML Prediction ---
    st.subheader("AQI Prediction Models")

    # Encode non-numeric columns
    for col in delhi_dataset.columns:
        if delhi_dataset[col].dtype == 'object':
            delhi_dataset[col] = LabelEncoder().fit_transform(delhi_dataset[col])

    X = delhi_dataset.drop(columns=['AQI'])
    y = delhi_dataset['AQI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Support Vector Regressor': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    accuracy_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred) * 100
        accuracy_scores[name] = round(score,2)

    st.write("### Model Accuracies (R² %)")
    st.table(accuracy_scores)

    # --- Bar chart for model comparison ---
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette='viridis')
    plt.xticks(rotation=30)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    st.pyplot(plt)

else:
    st.info("Please upload your CSV file to get started.")
