#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import os, time, json, base64, requests
import matplotlib.pyplot as plt
from datetime import datetime
from functools import lru_cache

# Optional maps
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


# -------------------------
# Config
# -------------------------
st.set_page_config(
    page_title="VISTA — Final Weather Dashboard",
    layout="wide",      
    initial_sidebar_state="collapsed"
)

PREDICTIONS_DIR = "predictions"
COORDS_FILE = "city_coords.json"
CLOUDS_IMAGE_PATH = "https://www.bing.com/images/search?view=detailV2&ccid=374n1XNw&id=A8C587E95CDF3A2ECD8DEE5BA544133357EE7CC6&thid=OIP.374n1XNwSoORB4FoEvK_mAAAAA&mediaurl=https%3a%2f%2fgiffiles.alphacoders.com%2f105%2f105240.gif&exph=253&expw=450&q=raindrops+moving+background+image+link&FORM=IRPRST&ck=C1DB95BFA5290CF0DFF4D5096B9706B5&selectedIndex=2&itb=0"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "VISTA-Weather-App/1.0"


# -------------------------
# Helpers
# -------------------------
def load_base64_image(path):
    return base64.b64encode(open(path, "rb").read()).decode() if os.path.exists(path) else ""

CLOUDS_IMG = load_base64_image(CLOUDS_IMAGE_PATH)


@st.cache_data(ttl=3600)
def list_prediction_cities(path):
    if not os.path.exists(path):
        return []
    return sorted(f.replace("_2075_predictions.csv", "")
                  for f in os.listdir(path) if f.endswith("_2075_predictions.csv"))


@st.cache_data(ttl=600)
def load_city_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    return df


def save_coords(data):
    with open(COORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@lru_cache(maxsize=256)
def geocode(city, pause=1.05):
    time.sleep(pause)
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"q": city, "format": "json", "limit": 1, "polygon_geojson": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=10
        )
        r.raise_for_status()
        d = r.json()
        return {
            "lat": float(d[0]["lat"]),
            "lon": float(d[0]["lon"]),
            "display_name": d[0].get("display_name"),
            "geojson": d[0].get("geojson")
        } if d else None
    except Exception:
        return None


# -------------------------
# Stored coords
# -------------------------
if os.path.exists(COORDS_FILE):
    try:
        city_coords = json.load(open(COORDS_FILE, "r", encoding="utf-8"))
    except Exception:
        city_coords = {}
else:
    city_coords = {}


# -------------------------
# CSS
# -------------------------
st.markdown(f"""
<style>
:root {{
  --primary:#006d77; --secondary:#83c5be;
  --background:#edf6f9; --text-dark:#003840; --text-light:#edf6f9;
}}
.stApp {{ background:var(--background); color:var(--text-dark); }}

.stApp::before {{
  content:""; position:fixed; top:0; left:-25%;
  width:200%; height:200%;
  background-image:url("data:image/png;base64,{CLOUDS_IMG}");
  opacity:.1; filter:hue-rotate(180deg) brightness(1.2);
  animation:moveClouds 120s linear infinite;
  pointer-events:none; z-index:0;
}}

@keyframes moveClouds {{
  from {{ transform:translateX(0%); }}
  to {{ transform:translateX(-50%); }}
}}

.vcard {{
  background:rgba(255,255,255,.95);
  border-left:6px solid var(--primary);
  border-radius:14px; padding:14px;
  box-shadow:0 8px 24px rgba(0,109,119,.12);
}}
.vcard.dark {{
  background:rgba(0,109,119,.35);
  color:var(--text-light);
  border-left-color:var(--secondary);
}}

.metric {{
  text-align:center; padding:12px;
  border-radius:12px; font-weight:700;
  background:var(--secondary); color:var(--primary);
}}

.footer {{ text-align:center; opacity:.7; font-size:13px; margin-top:20px; }}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Theme toggle
# -------------------------
st.session_state.setdefault("theme", "Auto")
choice = st.radio("Theme", ["Auto", "Light", "Dark"],
                  index=["Auto", "Light", "Dark"].index(st.session_state.theme))
st.session_state.theme = choice
DARK = choice == "Dark"


# -------------------------
# Header
# -------------------------
st.markdown(f"<div class='vcard{' dark' if DARK else ''}'>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1])
c1.title("🌤️ VISTA — AI Climate Dashboard")
c1.markdown("Explore projections (2025–2075)")
c2.markdown(f"*Theme:* {choice}")
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# City & data
# -------------------------
cities = list_prediction_cities(PREDICTIONS_DIR)
if not cities:
    st.error("No prediction files found.")
    st.stop()

city = st.selectbox("Select City", cities)
df = load_city_csv(os.path.join(PREDICTIONS_DIR, f"{city}_2075_predictions.csv"))

min_d, max_d = df["Date"].dt.date.min(), df["Date"].dt.date.max()
date = st.sidebar.date_input("Forecast date", min_d, min_d, max_d)
date_range = st.sidebar.date_input("Chart range", (min_d, max_d))
auto_coords = st.sidebar.checkbox("Auto-fetch coordinates", True)

row = df.iloc[(df["Date"].dt.date - date).abs().idxmin()]


# -------------------------
# Auto coords
# -------------------------
if city not in city_coords and auto_coords:
    r = geocode(city)
    if r:
        city_coords[city] = r
        save_coords(city_coords)


# -------------------------
# Info card
# -------------------------
st.markdown(f"<div class='vcard{' dark' if DARK else ''}'>", unsafe_allow_html=True)
st.subheader(f"{city} — {row['Date'].date()}")
cond = "⛈️" if row["Precipitation"] >= 5 else ("🔥" if row["Temperature"] >= 35 else "🌤️")
st.write(f"Condition: {cond}")
st.download_button("⬇ Download CSV", df.to_csv(index=False), f"{city}_2075_predictions.csv")
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Metrics
# -------------------------
st.markdown(f"<div class='vcard{' dark' if DARK else ''}'>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns([2,2,2,1])
m1.markdown(f"<div class='metric'>🌡 {row['Temperature']:.1f} °C</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric'>🌧 {row['Precipitation']:.2f} mm</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric'>💨 {row['WindSpeed']:.1f} km/h</div>", unsafe_allow_html=True)
m4.write(f"{min_d} → {max_d}")
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Charts
# -------------------------
st.markdown(f"<div class='vcard{' dark' if DARK else ''}'>", unsafe_allow_html=True)

if isinstance(date_range, tuple):
    dfp = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
else:
    dfp = df

for col, h in [("Temperature", "red"), ("Precipitation", "blue"), ("WindSpeed", "green")]:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(dfp["Date"], dfp[col])
    ax.axvline(row["Date"], linestyle="--")
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Risk computation (unchanged)
# -------------------------
def compute_risk(df):
    try:
        x = (df["Date"] - df["Date"].min()).dt.days.values
        y = df["Temperature"].values
        m = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0][0]
        temp_dec = m * 365 * 10

        early, late = df[df["Year"]<=2034], df[df["Year"]>=2066]
        precip_pct = ((late["Precipitation"].mean() - early["Precipitation"].mean())
                       / max(early["Precipitation"].mean(), 1e-6) * 100)

        ext_p = (df["Precipitation"] >= 10).mean()
        ext_w = (df["WindSpeed"] >= 20).mean()

        score = (
            0.45*np.interp(temp_dec, [-1,0,1.5,3], [0,5,70,100]) +
            0.35*np.interp(precip_pct, [-50,0,100], [0,10,100]) +
            0.2*np.interp(ext_w, [0,.05,.2], [0,25,80])
        )
        score = float(np.clip(score, 0, 100))
        cat = "Low" if score<33 else ("Medium" if score<66 else "High")
        return score, cat, temp_dec, precip_pct, ext_p, ext_w
    except Exception:
        return None, "Unknown", None, None, None, None


score, cat, td, pp, ep, ew = compute_risk(df)


# -------------------------
# Risk card
# -------------------------
st.markdown(f"<div class='vcard{' dark' if DARK else ''}'>", unsafe_allow_html=True)
st.subheader("🔎 Climate Risk Summary")
if score is None:
    st.warning("Risk could not be computed.")
else:
    c1, c2 = st.columns([1,2])
    c1.metric(f"Risk: {cat}", f"{score:.1f}/100")
    c2.write(f"Temp trend: {td:.2f} °C/decade")
    c2.write(f"Precip change: {pp:.2f}%")
    c2.write(f"Extreme precip: {ep:.2f}")
    c2.write(f"Extreme wind: {ew:.2f}")
st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Map
# -------------------------
st.markdown(f"<div class='vcard{' dark' if DARK else ''}'>", unsafe_allow_html=True)
coords = city_coords.get(city)

if coords and FOLIUM_AVAILABLE:
    m = folium.Map([coords["lat"], coords["lon"]], zoom_start=10)
    folium.Marker([coords["lat"], coords["lon"]], popup=city).add_to(m)
    st_folium(m, width=900, height=450)
elif coords:
    st.map(pd.DataFrame([coords]))

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Footer
# -------------------------
st.markdown("<div class='footer'>🌍 VISTA — Final Dashboard • Built with Streamlit</div>",
            unsafe_allow_html=True)