"""
SafeDrive Predictor — Streamlit Frontend (US Imperial units)
Run: streamlit run frontend/app.py
(Make sure the FastAPI backend is running at http://localhost:8080)
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

API_BASE = "http://localhost:8080"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeDrive Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0f1117; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2d3250;
        margin-bottom: 1rem;
    }
    .rec-critical { border-left: 4px solid #ff4b4b; background: #2a1515; padding: 0.8rem 1rem; border-radius:6px; margin:6px 0; }
    .rec-high     { border-left: 4px solid #ffa500; background: #2a1f10; padding: 0.8rem 1rem; border-radius:6px; margin:6px 0; }
    .rec-medium   { border-left: 4px solid #f9d71c; background: #2a2710; padding: 0.8rem 1rem; border-radius:6px; margin:6px 0; }
    .rec-low      { border-left: 4px solid #4caf50; background: #122212; padding: 0.8rem 1rem; border-radius:6px; margin:6px 0; }
    .rec-info     { border-left: 4px solid #4fc3f7; background: #10202a; padding: 0.8rem 1rem; border-radius:6px; margin:6px 0; }
    h1, h2, h3 { color: #e8eaf6 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def make_gauge(score: float, level: str, color: str) -> go.Figure:
    color_map = {"green": "#4caf50", "yellow": "#f9d71c", "orange": "#ffa500", "red": "#ff4b4b"}
    hex_color = color_map.get(color, "#4fc3f7")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50, "increasing": {"color": "#ff4b4b"}, "decreasing": {"color": "#4caf50"}},
        title={"text": f"Risk Level: <b>{level}</b>", "font": {"size": 20, "color": "#e8eaf6"}},
        number={"font": {"size": 52, "color": hex_color}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555", "tickfont": {"color": "#aaa"}},
            "bar": {"color": hex_color, "thickness": 0.25},
            "bgcolor": "#1e2130",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#1a2e1a"},
                {"range": [30, 55], "color": "#2e2a10"},
                {"range": [55, 75], "color": "#2e1e0a"},
                {"range": [75,100], "color": "#2e0f0f"},
            ],
            "threshold": {
                "line": {"color": hex_color, "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font={"color": "#e8eaf6"},
        height=320,
        margin=dict(l=30, r=30, t=20, b=10),
    )
    return fig


def make_scenario_chart(base_payload: dict, current_precip: float) -> go.Figure:
    """Risk score across a precipitation range (in/hr)."""
    precip_range = [round(p * 0.1, 1) for p in range(0, 21)]  # 0.0 to 2.0 in/hr
    scores = []
    for p in precip_range:
        payload = {**base_payload, "scenario_precipitation": p, "city": None}
        result = api_post("/predict", payload)
        scores.append(result["risk_score"] if result else None)

    df = pd.DataFrame({"Precipitation (in/hr)": precip_range, "Risk Score": scores})
    df = df.dropna()

    fig = px.area(
        df, x="Precipitation (in/hr)", y="Risk Score",
        title="Risk Score vs. Precipitation Scenario",
        color_discrete_sequence=["#4fc3f7"],
    )
    fig.add_vline(
        x=current_precip, line_dash="dash", line_color="#ffa500",
        annotation_text=f"Now: {current_precip:.2f} in/hr",
        annotation_font_color="#ffa500",
    )
    fig.add_hrect(y0=0,  y1=30, fillcolor="#4caf50", opacity=0.07, line_width=0)
    fig.add_hrect(y0=30, y1=55, fillcolor="#f9d71c", opacity=0.07, line_width=0)
    fig.add_hrect(y0=55, y1=75, fillcolor="#ffa500", opacity=0.07, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#ff4b4b", opacity=0.07, line_width=0)
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font={"color": "#e8eaf6"}, height=280,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def render_recommendations(recs: list):
    css_map = {
        "CRITICAL": "rec-critical",
        "HIGH":     "rec-high",
        "MEDIUM":   "rec-medium",
        "LOW":      "rec-low",
        "INFO":     "rec-info",
    }
    for rec in recs:
        css_class = css_map.get(rec.get("priority", "INFO"), "rec-info")
        icon  = rec.get("icon", "ℹ️")
        title = rec.get("title", "")
        body  = rec.get("body", "")
        st.markdown(
            f'<div class="{css_class}"><strong>{icon} {title}</strong><br>'
            f'<span style="color:#ccc;font-size:0.9em">{body}</span></div>',
            unsafe_allow_html=True,
        )


def feature_importance_chart(importances: dict) -> go.Figure:
    labels = list(importances.keys())
    values = list(importances.values())
    sorted_pairs = sorted(zip(values, labels), reverse=True)
    values_s, labels_s = zip(*sorted_pairs)
    fig = go.Figure(go.Bar(
        x=list(values_s), y=list(labels_s), orientation="h",
        marker_color="#4fc3f7",
    ))
    fig.update_layout(
        title="Model Feature Importances",
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2130",
        font={"color": "#e8eaf6"}, height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Importance", yaxis_title="",
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Inputs
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-door-lock.png", width=64)
    st.title("SafeDrive Predictor")
    st.caption("Real-time road risk powered by ML + live data")
    st.divider()

    st.header("📍 Location")
    city = st.text_input("City (for live weather)", placeholder="e.g. Chicago, Houston", value="")

    st.header("🚗 Vehicle")
    vin_input = st.text_input(
        "VIN (optional, 17 chars)",
        placeholder="e.g. 1HGCM82633A004352",
        max_chars=17,
    ).strip().upper()

    st.header("👤 Driver Profile")
    driver_id = st.text_input("Driver ID (any name/ID)", value="driver_01")
    accidents = st.slider("Accidents (last 3 yrs)", 0, 10, 0)
    tickets   = st.slider("Tickets (last 3 yrs)",   0, 10, 0)

    st.header("🌦️ Manual Weather Override")
    st.caption("Used if no city entered, or for what-if testing")
    manual_temp   = st.slider("Temperature (°F)",        -20, 120, 59)
    manual_precip = st.slider("Precipitation (in/hr)",   0.0, 2.0, 0.0, step=0.01, format="%.2f")
    manual_vis    = st.slider("Visibility (miles)",       0.1, 6.2, 6.2, step=0.1,  format="%.1f")

    st.header("⏰ Time")
    use_current_time = st.checkbox("Use current hour", value=True)
    manual_hour = st.slider("Hour of day (0–23)", 0, 23, datetime.now().hour)
    hour_of_day = datetime.now().hour if use_current_time else manual_hour

    st.divider()
    predict_btn = st.button("🔍 Calculate Risk", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════

st.markdown("# 🛡️ SafeDrive Predictor")
st.markdown("*Composite road risk scoring using live weather, NHTSA recall data, and machine learning.*")

# ── VIN Lookup panel ─────────────────────────────────────────────────────────
if vin_input:
    with st.expander("🔎 VIN Decoder Results", expanded=True):
        with st.spinner("Querying NHTSA vPIC API..."):
            vin_data = api_get(f"/vin/{vin_input}")
        if vin_data:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Make",  vin_data.get("make",  "—"))
            c2.metric("Model", vin_data.get("model", "—"))
            c3.metric("Year",  vin_data.get("year",  "—"))
            recall_count = vin_data.get("recall_count", 0)
            c4.metric(
                "Active Recalls", recall_count,
                delta="Action needed" if recall_count else None,
                delta_color="inverse",
            )
            if vin_data.get("has_recall"):
                st.error("🚨 **This vehicle has active safety recalls!**")
                for rec in vin_data.get("recalls", [])[:5]:
                    st.markdown(
                        f"**Component:** {rec.get('component', 'N/A')}  \n"
                        f"**Summary:** {rec.get('summary', 'N/A')}"
                    )
            else:
                st.success("✅ No active NHTSA recalls found for this vehicle.")

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    payload = {
        "city": city if city else None,
        "temperature":   manual_temp,
        "precipitation": manual_precip,
        "visibility":    manual_vis,
        "driver_id":     driver_id or "anonymous",
        "accidents_last_3yr": accidents,
        "tickets_last_3yr":   tickets,
        "vin":       vin_input if vin_input else None,
        "hour_of_day": hour_of_day,
    }

    with st.spinner("Calculating risk..."):
        result = api_post("/predict", payload)

    if result:
        score    = result["risk_score"]
        level    = result["risk_level"]
        color    = result["risk_color"]
        weather  = result.get("weather")
        vehicle  = result.get("vehicle")
        recs     = result.get("recommendations", [])
        multipliers = result.get("applied_multipliers", [])
        importances = result.get("feature_importances", {})

        # ── Row 1: Gauge + Conditions ─────────────────────────────────────
        col_gauge, col_stats = st.columns([1.4, 1], gap="large")

        with col_gauge:
            st.plotly_chart(make_gauge(score, level, color), use_container_width=True)
            if multipliers:
                st.markdown("**Active risk multipliers:**")
                for m in multipliers:
                    st.markdown(f"  - `{m}`")

        with col_stats:
            st.markdown("### Conditions Summary")
            if weather:
                wcol1, wcol2 = st.columns(2)
                wcol1.metric("🌡️ Temp",          f"{weather['temperature']}°F")
                wcol2.metric("🌧️ Precipitation", f"{weather['precipitation']:.2f} in/hr")
                wcol1.metric("👁️ Visibility",    f"{weather['visibility']:.1f} mi")
                wcol2.metric("💨 Wind",           f"{weather['wind_speed']:.0f} mph")
                wcol1.metric("💧 Humidity",       f"{weather['humidity']}%")
                wcol2.metric("☁️ Conditions",     weather.get("description", "—"))
            else:
                st.metric("Temp",          f"{manual_temp}°F")
                st.metric("Precipitation", f"{manual_precip:.2f} in/hr")
                st.metric("Visibility",    f"{manual_vis:.1f} mi")

            st.markdown("---")
            st.metric("Base ML Score", f"{result['base_score']:.1f} / 100")
            st.metric(
                "Final Score", f"{score:.1f} / 100",
                delta=f"{score - result['base_score']:+.1f} from weighting",
            )

            if vehicle:
                st.markdown("---")
                st.markdown(f"**Vehicle:** {vehicle['year']} {vehicle['make']} {vehicle['model']}")
                recall_badge = "🚨 Has Recall" if vehicle["has_recall"] else "✅ No Recalls"
                st.markdown(f"**Recall status:** {recall_badge}")

        st.divider()

        # ── Row 2: Scenario chart + Feature importances ───────────────────
        col_scenario, col_import = st.columns(2, gap="large")

        with col_scenario:
            st.markdown("### 🎚️ What-If: Scenario Slider")
            current_p = float(weather["precipitation"]) if weather else manual_precip
            current_v = float(weather["visibility"]) if weather else manual_vis

            scenario_precip = st.slider(
                "Simulate Precipitation (in/hr)", 0.0, 2.0, current_p,
                step=0.01, format="%.2f", key="sc_precip",
            )
            scenario_vis = st.slider(
                "Simulate Visibility (miles)", 0.1, 6.2, current_v,
                step=0.1, format="%.1f", key="sc_vis",
            )
            if st.button("Run Scenario", key="run_scenario"):
                sc_payload = {
                    **payload,
                    "scenario_precipitation": scenario_precip,
                    "scenario_visibility":    scenario_vis,
                }
                with st.spinner("Running scenario..."):
                    sc_result = api_post("/predict", sc_payload)
                if sc_result:
                    sc_score = sc_result["risk_score"]
                    sc_level = sc_result["risk_level"]
                    delta_val = sc_score - score
                    st.metric(
                        f"Scenario Risk ({sc_level})",
                        f"{sc_score:.1f} / 100",
                        delta=f"{delta_val:+.1f} vs current",
                        delta_color="inverse",
                    )

            base_payload_sc = {
                "temperature":        float(weather["temperature"]) if weather else manual_temp,
                "precipitation":      current_p,
                "visibility":         current_v,
                "accidents_last_3yr": accidents,
                "tickets_last_3yr":   tickets,
                "hour_of_day":        hour_of_day,
                "vin":                vin_input or None,
            }
            with st.spinner("Building scenario chart..."):
                st.plotly_chart(
                    make_scenario_chart(base_payload_sc, current_p),
                    use_container_width=True,
                )

        with col_import:
            st.markdown("### 🤖 Model Feature Importances")
            if importances:
                st.plotly_chart(feature_importance_chart(importances), use_container_width=True)
            else:
                st.info("Feature importances not available.")

        st.divider()

        # ── Row 3: Safety Recommendations ────────────────────────────────
        st.markdown("### 🔔 Safety Recommendations")
        if recs:
            render_recommendations(recs)
        else:
            st.success("No specific warnings for current conditions. Drive safely!")

else:
    # ── Landing state — show history ──────────────────────────────────────
    st.info("👈 Fill in your details in the sidebar and click **Calculate Risk** to get started.")

    st.markdown("### 📊 Recent Predictions")
    history = api_get("/history?limit=15")
    if history:
        df = pd.DataFrame(history)
        df = df[["created_at", "driver_id", "vin", "final_score", "risk_level",
                 "precipitation", "visibility", "has_recall"]]
        df.columns = ["Time", "Driver", "VIN", "Score", "Level",
                      "Precip (in/hr)", "Visibility (mi)", "Recall"]
        df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            df.style.background_gradient(subset=["Score"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("No prediction history yet. Run your first prediction!")
