#!/usr/bin/env python3
"""
Weather Data Mine (Open-Meteo, no API key)

- Single-file Streamlit app for near real-time weather around Grand Junction, CO.
- Fetches hourly data from Open-Meteo, writes CSV cache to data/ (git-ignored).
- Shows recent samples, charts, daily summaries, and a rolling z-score anomaly.

Assignment-friendly:
- Only the toolchain (this file) is committed.
- Raw data saved locally under data/ and excluded via .gitignore.

Run:
  streamlit run app_streamlit.py
"""

import os
import json
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------- Config ----------------------------
# Default: Grand Junction, CO (approx)
DEFAULT_LOCATIONS = {
    "Grand Junction, CO": (39.0639, -108.5506),
    "Fruita, CO": (39.1589, -108.7280),
    "Palisade, CO": (39.1108, -108.3509),
    "Montrose, CO": (38.4783, -107.8762),
}

LOCAL_TZ = "America/Denver"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------- Helpers ----------------------------
def open_meteo_url(lat: float, lon: float, past_hours: int) -> str:
    from urllib.parse import urlencode
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
        "past_hours": past_hours,     # go back N hours from *now*
        "forecast_hours": 0,          # only past window
        "wind_speed_unit": "ms",      # so our chart label (m/s) is correct
        "precipitation_unit": "mm",   # default, but explicit
        "temperature_unit": "fahrenheit"
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
    # 'time' comes as ISO strings in UTC
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.set_index("time").sort_index()
    # Make sure numeric cols are numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_local_naive(df_utc_idx: pd.DataFrame) -> pd.DataFrame:
    """UTC DatetimeIndex → convert to local tz then drop tz for Streamlit/Arrow friendliness."""
    if df_utc_idx.empty:
        return df_utc_idx
    out = df_utc_idx.copy()
    out.index = out.index.tz_convert(LOCAL_TZ).tz_localize(None)
    return out

def daily_summary(df_local_naive: pd.DataFrame) -> pd.DataFrame:
    """Simple daily stats from hourly: mean/min/max temp, precip sum, wind mean."""
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
    """Z-score vs rolling mean/std (e.g., 24 hours)."""
    if s is None or s.empty:
        return s
    mu = s.rolling(window, min_periods=max(4, window//4)).mean()
    sd = s.rolling(window, min_periods=max(4, window//4)).std()
    return (s - mu) / sd

def cache_path(lat: float, lon: float) -> str:
    return os.path.join(DATA_DIR, f"weather_{lat:.4f}_{lon:.4f}.csv")

def save_cache(df_utc: pd.DataFrame, lat: float, lon: float) -> None:
    # Write UTC index to CSV; keep it UTC in cache
    path = cache_path(lat, lon)
    df_utc.to_csv(path)

def load_cache(lat: float, lon: float) -> pd.DataFrame:
    path = cache_path(lat, lon)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    # Ensure index is UTC tz-aware for consistent conversions later
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframes are friendly to Streamlit's Arrow serializer."""
    if df.empty:
        return df
    frame = df.copy()
    frame = frame.reset_index()
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

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Grand Junction Weather Mine", layout="wide")
st.title("Near Real-Time Weather — Western Colorado (Open-Meteo)")

# Sidebar controls
loc_label = st.sidebar.selectbox("Location", list(DEFAULT_LOCATIONS.keys()), index=0)
lat, lon = DEFAULT_LOCATIONS[loc_label]
days = st.sidebar.slider("Window (days)", 1, 14, 7)
show_debug = st.sidebar.checkbox("Show raw query URL", value=False)

# Fetch (fresh each run) and cache
try:
    js = fetch_hourly_json(lat, lon, days)
    df_utc = hourly_to_frame(js)
    # Save/replace cache each run (simple, reproducible)
    save_cache(df_utc, lat, lon)
except Exception as e:
    st.error(f"Fetch error: {e}")
    df_utc = load_cache(lat, lon)
    if df_utc.empty:
        st.stop()

if show_debug and "_query_url" in js:
    st.caption(f"Query: {js['_query_url']}")

# Convert to local time (America/Denver), then drop tz for display
df_local = to_local_naive(df_utc)

# Layout
left, right = st.columns(2)

with left:
    st.subheader("Recent hourly samples")
    st.dataframe(arrow_safe_df(df_local.tail(24)), use_container_width=True)

    # Temperature chart
    if "temperature_2m" in df_local.columns:
        chart_temp = (
            alt.Chart(arrow_safe_df(df_local.reset_index().rename(columns={"time":"t"})))
            .mark_line()
            .encode(x="t:T", y=alt.Y("temperature_2m:Q", title="Temperature (°C)"))
            .properties(height=220)
        )
        st.altair_chart(chart_temp, use_container_width=True)

    # Wind speed
    if "wind_speed_10m" in df_local.columns:
        chart_wind = (
            alt.Chart(arrow_safe_df(df_local.reset_index().rename(columns={"time":"t"})))
            .mark_line()
            .encode(x="t:T", y=alt.Y("wind_speed_10m:Q", title="Wind Speed (m/s)"))
            .properties(height=220)
        )
        st.altair_chart(chart_wind, use_container_width=True)

with right:
    st.subheader("Daily summaries & simple anomaly")

    # Daily summary table
    daily = daily_summary(df_local)
    if daily.empty:
        st.info("No daily data yet.")
    else:
        st.markdown("**Daily summary (last few days)**")
        st.dataframe(arrow_safe_df(daily.tail(7)), use_container_width=True)

        # Rolling anomaly (z-score) on temperature as a simple example
        if "temperature_2m" in df_local.columns:
            df_anom = df_local[["temperature_2m"]].copy()
            df_anom["temp_z24"] = rolling_anomaly(df_anom["temperature_2m"], window=24)
            zdf = arrow_safe_df(df_anom.reset_index().rename(columns={"time": "t"}))
            chart_z = (
                alt.Chart(zdf)
                .mark_line()
                .encode(x="t:T", y=alt.Y("temp_z24:Q", title="Temp z-score (24h rolling)"))
                .properties(height=220)
            )
            st.altair_chart(chart_z, use_container_width=True)
        else:
            st.info("No temperature column found for anomaly calculation.")

st.caption("Source: Open-Meteo hourly forecast/analysis API (no API key). Time converted to America/Denver.")
