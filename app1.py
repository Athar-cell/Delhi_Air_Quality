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


