#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

st.set_page_config(page_title="AI Weather Predictor (2025–2075)", layout="centered")

st.title("AI Weather Predictor (2025–2075)")
st.markdown("""
This app uses an AI model trained on historical data (1980–2024)  
to forecast daily **temperature**, **precipitation**, and **wind speed**  
for each city up to the year **2075**.
""")

predictions_dir = "predictions"

if not os.path.exists(predictions_dir):
    st.error("Predictions folder not found! Please train the model first.")
    st.stop()

city_files = [f for f in os.listdir(predictions_dir) if f.endswith("_2075_predictions.csv")]

if not city_files:
    st.error("No prediction files found. Please ensure your training script created *_2075_predictions.csv files.")
    st.stop()

cities = [f.replace("_2075_predictions.csv", "") for f in city_files]

selected_city = st.selectbox("Select a City", cities)

file_path = os.path.join(predictions_dir, f"{selected_city}_2075_predictions.csv")
city_df = pd.read_csv(file_path)
city_df["Date"] = pd.to_datetime(city_df["Date"])

min_date = city_df["Date"].min().date()
max_date = city_df["Date"].max().date()

st.markdown(f"Available prediction range: **{min_date} → {max_date}**")

user_date = st.date_input(
    "Select a date between 2025 and 2075:",
    min_value=min_date,
    max_value=max_date,
    value=min_date
)

selected_row = city_df[city_df["Date"] == pd.Timestamp(user_date)]

if selected_row.empty:
    st.warning("No prediction available for this exact date (try a nearby day).")
else:
    st.subheader(f"Weather Forecast for {selected_city} on {user_date}:")
    st.write(f"**Temperature:** {selected_row['Temperature'].values[0]:.2f} °C")
    st.write(f"**Precipitation:** {selected_row['Precipitation'].values[0]:.2f} mm")
    st.write(f"**Wind Speed:** {selected_row['WindSpeed'].values[0]:.2f} km/h")

st.markdown("---")
st.subheader(f"📈 Long-Term Weather Trends for {selected_city}")

feature = st.selectbox("Select Feature to Plot", ["Temperature", "Precipitation", "WindSpeed"])

years = ["All Years"] + sorted(list(city_df["Date"].dt.year.unique()))
selected_year = st.selectbox("📅 Select Year to Display", years)

if selected_year != "All Years":
    plot_df = city_df[city_df["Date"].dt.year == int(selected_year)]
    title_year_range = f"{selected_year}"
else:
    plot_df = city_df.copy()
    title_year_range = "2025–2075"

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["Date"], plot_df[feature], label=f"{feature} ({title_year_range})")
ax.set_title(f"{feature} Forecast for {selected_city} ({title_year_range})")
ax.set_xlabel("Date")
ax.set_ylabel(feature)
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
csv = city_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Full Forecast Data (CSV)",
    data=csv,
    file_name=f"{selected_city}_2075_predictions.csv",
    mime="text/csv"
)

st.success("Forecasts loaded successfully!")
