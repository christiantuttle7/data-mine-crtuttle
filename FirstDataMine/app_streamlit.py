# app_streamlit.py
#!/usr/bin/env python3

import altair as alt
import pandas as pd
import streamlit as st

from config import DEFAULT_LOCATIONS, LOCAL_TZ
from weather_core import (
    fetch_hourly_json, hourly_to_frame, to_local_naive,
    daily_summary, rolling_anomaly, save_cache, load_cache, arrow_safe_df
)

st.set_page_config(page_title="Grand Junction Weather Data", layout="wide")
st.title("Real Time Weather - Grand Junction")

# Sidebar
loc_label = st.sidebar.selectbox("Location", list(DEFAULT_LOCATIONS.keys()), index=0)
lat, lon = DEFAULT_LOCATIONS[loc_label]
days = st.sidebar.slider("Window (days)", 1, 14, 7)
show_debug = st.sidebar.checkbox("Show raw query URL", value=False)
temp_unit_label = "°F"  # keep in sync with weather_core.open_meteo_url()

# Fetch + cache (logic is in core)
try:
    js = fetch_hourly_json(lat, lon, days)
    df_utc = hourly_to_frame(js)
    save_cache(df_utc, lat, lon)
except Exception as e:
    st.error(f"Fetch error: {e}")
    df_utc = load_cache(lat, lon)
    if df_utc.empty():
        st.stop()

if show_debug and isinstance(js, dict) and "_query_url" in js:
    st.caption(f"Query: {js['_query_url']}")

df_local = to_local_naive(df_utc)

left, right = st.columns(2)

with left:
    st.subheader("Recent hourly samples")
    st.dataframe(arrow_safe_df(df_local.tail(24)), use_container_width=True)

    if "temperature_2m" in df_local.columns:
        temp_df = arrow_safe_df(df_local.reset_index().rename(columns={"time": "t"}))
        chart_temp = (
            alt.Chart(temp_df)
            .mark_line()
            .encode(x="t:T", y=alt.Y("temperature_2m:Q", title=f"Temperature ({temp_unit_label})"))
            .properties(height=220)
        )
        st.altair_chart(chart_temp, use_container_width=True)

    if "wind_speed_10m" in df_local.columns:
        wind_df = arrow_safe_df(df_local.reset_index().rename(columns={"time": "t"}))
        chart_wind = (
            alt.Chart(wind_df)
            .mark_line()
            .encode(x="t:T", y=alt.Y("wind_speed_10m:Q", title="Wind Speed (m/s)"))
            .properties(height=220)
        )
        st.altair_chart(chart_wind, use_container_width=True)

with right:
    st.subheader("Daily summaries & simple anomaly")

    daily = daily_summary(df_local)
    if daily.empty:
        st.info("No daily data yet.")
    else:
        st.markdown("**Daily summary (last few days)**")
        st.dataframe(arrow_safe_df(daily.tail(7)), use_container_width=True)

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

st.caption(f"Source: Open-Meteo hourly API (no API key). Times converted to {LOCAL_TZ}.")
