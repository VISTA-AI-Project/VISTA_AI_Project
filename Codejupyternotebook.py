import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tkinter import Tk, filedialog
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
tf.random.set_seed(42)

root = Tk()
root.title("Select your CSV file")
root.geometry("0x0")

file_path = filedialog.askopenfilename(
    title="Select weather_data.csv",
    filetypes=[("CSV files", "*.csv")]
)
root.destroy()

if not file_path:
    raise FileNotFoundError(" No file selected! Please select your CSV file.")

print(f" File selected: {file_path}")


df = pd.read_csv(file_path)
df.columns = ['City', 'Date', 'Temperature', 'Precipitation', 'WindSpeed']
df['Date'] = pd.to_datetime(df['Date'])

print(f" Loaded dataset with {len(df)} rows and {len(df['City'].unique())} cities.")

cities = df['City'].unique().tolist()
print(f" Cities detected: {cities}")


features = ["Temperature", "Precipitation", "WindSpeed"]
seq_length = 60
future_days = 50 * 365 

os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

resume = True  


print("\n Starting training for all cities...\n")

for city in tqdm(cities, desc="Training progress", ncols=100):
    try:
        model_path = f"models/{city}_daily_model.keras"
        pred_path = f"predictions/{city}_2075_predictions.csv"

        if resume and (os.path.exists(model_path) or os.path.exists(pred_path)):
            print(f"Skipping {city}: Model or predictions already exist.")
            continue

        print(f"\n Training LSTM for {city}...")

        city_df = df[df['City'] == city].sort_values('Date').reset_index(drop=True)

        if len(city_df) < seq_length + 1:
            print(f" Skipping {city}: Not enough data ({len(city_df)} records).")
            continue

        city_df = city_df.set_index('Date').resample('D').interpolate().reset_index()

        city_df = city_df.dropna(subset=features)

        city_df["DayOfYear"] = city_df["Date"].dt.dayofyear
        city_df["Sin_Day"] = np.sin(2 * np.pi * city_df["DayOfYear"] / 365)
        city_df["Cos_Day"] = np.cos(2 * np.pi * city_df["DayOfYear"] / 365)

        features_with_season = features + ["Sin_Day", "Cos_Day"]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(city_df[features_with_season])

        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i : i + seq_length])
            y.append(scaled_data[i + seq_length])
        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            print(f" No usable sequences for {city}, skipping.")
            continue

        print(f" Training data for {city}: X={X.shape}, y={y.shape}")

        tf.keras.backend.clear_session()

       
        model = Sequential([
            LSTM(96, return_sequences=True, input_shape=(seq_length, len(features_with_season))),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(len(features_with_season))
        ])

        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        print(f" Training model for {city}...")
        model.fit(X, y, epochs=50, batch_size=32, verbose=1, callbacks=[es])

        model.save(model_path)
        print(f"Model saved for {city}")

       
        print(f" Generating 50-year predictions for {city}...")
        last_seq = scaled_data[-seq_length:].copy()
        scaled_preds = []

        tiny_scaled_noise_sd = 0.002

        for day in range(future_days):
            pred_scaled = model.predict(np.expand_dims(last_seq, axis=0), verbose=0)[0]

            # small jitter in scaled space so the recursive sequence doesn't fully collapse
            pred_scaled += np.random.normal(0, tiny_scaled_noise_sd, size=pred_scaled.shape)

            scaled_preds.append(pred_scaled)
            last_seq = np.vstack((last_seq[1:], pred_scaled))

        scaled_preds = np.array(scaled_preds)  # shape (future_days, len(features_with_season))

        preds_phys = scaler.inverse_transform(scaled_preds)  # now in original units for all columns

        days = np.arange(future_days)

        temp_season_amp = 6.0     # degrees C amplitude (annual)
        precip_season_amp = 5.0   # mm amplitude (semi-annual-ish)
        wind_season_amp = 2.0     # km/h amplitude

        seasonal_phys = np.vstack([
            temp_season_amp * np.sin(2 * np.pi * (days % 365) / 365),
            precip_season_amp * np.sin(2 * np.pi * (days % 180) / 180),
            wind_season_amp * np.sin(2 * np.pi * (days % 90) / 90)
        ]).T  # shape (future_days, 3)

        noise_phys = np.vstack([
            np.random.normal(0, 0.5, future_days),   # temp noise sd = 0.5°C
            np.random.normal(0, 1.0, future_days),   # precip noise sd = 1.0 mm
            np.random.normal(0, 0.8, future_days)    # wind noise sd = 0.8 km/h
        ]).T

        preds_phys[:, 0:3] = preds_phys[:, 0:3] + seasonal_phys + noise_phys

        
        preds_phys[:, 0] = np.clip(preds_phys[:, 0], -30.0, 50.0)

        preds_phys[:, 1] = np.clip(preds_phys[:, 1], 0.0, 500.0)

        preds_phys[:, 2] = np.clip(preds_phys[:, 2], 0.0, 200.0)

       
        future_dates = pd.date_range(start="2025-01-01", periods=future_days, freq="D")

        city_preds = pd.DataFrame(preds_phys, columns=features_with_season)
        city_preds["Date"] = future_dates
        city_preds["City"] = city
        city_preds = city_preds[["City", "Date"] + features]

        city_preds.to_csv(pred_path, index=False)
        print(f"Predictions saved for {city}: {pred_path}")

        del model, X, y, city_df, scaled_preds, scaled_data, preds_phys
        gc.collect()
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f" Error training {city}: {e}")
        continue


combined_path = "predictions/All_Cities_2075_Predictions.csv"
all_city_files = [f for f in os.listdir("predictions") if f.endswith("_2075_predictions.csv")]

combined_df = pd.concat([
    pd.read_csv(os.path.join("predictions", f))
    for f in all_city_files
], ignore_index=True)

combined_df.to_csv(combined_path, index=False)

print(f"\n All city predictions saved at {combined_path} (2025–2075)")
