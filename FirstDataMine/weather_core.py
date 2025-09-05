# weather_core.py
import os
from urllib.parse import urlencode
from datetime import timezone
import requests
import pandas as pd
from config import DATA_DIR, LOCAL_TZ

"""
Name: Christian Tuttle
Date: 9/4/2025
Project: Automate Data Source
Description: This program takes in weather data for Grand Junction, including temperature, humidity, 
precipitation, etc. It creates a streamlit app, lets the user choose a time window (within the past 14 days)
and provides tables and graphs. 
Note: This follows very closely to the colorado river example. 
Im guessing that is ok based off the assignment description

"""

def open_meteo_url(lat: float, lon: float, past_hours: int) -> str:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
        ]),
        "timezone": "UTC",
        "past_hours": past_hours,
        "forecast_hours": 0,
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        # NOTE: choose 'celsius' or 'fahrenheit' here; document in UI labels.
        "temperature_unit": "fahrenheit",
    }
    return f"{base}?{urlencode(params)}"

def fetch_hourly_json(lat: float, lon: float, days: int) -> dict:
    url = open_meteo_url(lat, lon, past_hours=days * 24)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    js["_query_url"] = url
    return js

def hourly_to_frame(js: dict) -> pd.DataFrame:
    hourly = js.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.set_index("time").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_local_naive(df_utc_idx: pd.DataFrame) -> pd.DataFrame:
    if df_utc_idx.empty:
        return df_utc_idx
    out = df_utc_idx.copy()
    out.index = out.index.tz_convert(LOCAL_TZ).tz_localize(None)
    return out

def daily_summary(df_local_naive: pd.DataFrame) -> pd.DataFrame:
    if df_local_naive.empty:
        return pd.DataFrame()
    daily = df_local_naive.resample("D").agg({
        "temperature_2m": ["mean", "min", "max"],
        "precipitation": "sum",
        "wind_speed_10m": "mean",
    })
    daily.columns = ["_".join(col).strip("_") for col in daily.columns.to_flat_index()]
    return daily

def rolling_anomaly(s: pd.Series, window: int = 24) -> pd.Series:
    if s is None or s.empty:
        return s
    mu = s.rolling(window, min_periods=max(4, window//4)).mean()
    sd = s.rolling(window, min_periods=max(4, window//4)).std()
    return (s - mu) / sd

def cache_path(lat: float, lon: float) -> str:
    return os.path.join(DATA_DIR, f"weather_{lat:.4f}_{lon:.4f}.csv")

def save_cache(df_utc: pd.DataFrame, lat: float, lon: float) -> None:
    df_utc.to_csv(cache_path(lat, lon))

def load_cache(lat: float, lon: float) -> pd.DataFrame:
    path = cache_path(lat, lon)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.reset_index().copy()
    for c in frame.columns:
        if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_string_dtype(frame[c]):
            try:
                coerced = pd.to_datetime(frame[c], errors="coerce")
                if coerced.notna().sum() >= max(1, int(0.8 * len(coerced))):
                    frame[c] = coerced
            except Exception:
                pass
        if pd.api.types.is_datetime64_any_dtype(frame[c]):
            try:
                if getattr(frame[c].dt, "tz", None) is not None:
                    frame[c] = frame[c].dt.tz_localize(None)
            except Exception:
                frame[c] = frame[c].astype(str)
    return frame
