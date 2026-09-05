import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="Industrial AI Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS
st.markdown("""
<style>
    /* Global Styles & Dark Theme */
    .stApp {
        background-color: #080b11;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Top Brand Header */
    .brand-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(8, 11, 17, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 242, 254, 0.15);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.75rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Status Badges */
    .badge-ok {
        background: rgba(0, 230, 118, 0.12);
        border: 1px solid rgba(0, 230, 118, 0.3);
        color: #00e676;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-danger {
        background: rgba(255, 23, 68, 0.15);
        border: 1px solid rgba(255, 23, 68, 0.5);
        color: #ff1744;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        animation: pulse-red 1.5s infinite;
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.6); }
        70% { box-shadow: 0 0 0 10px rgba(255, 23, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }
    }

    /* AI Insight Box */
    .ai-box {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #00f2fe;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .ai-box-danger {
        background: rgba(30, 10, 20, 0.85);
        border-left: 4px solid #ff1744;
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 10px;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.2);
    }

    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #05070c;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "Timestamp", "Asset", "Vibration", "Temperature", "Current", "Power", "Anomaly_Score", "Status"
    ])

if "fault_mode" not in st.session_state:
    st.session_state.fault_mode = "NORMAL"

if "work_order_triggered" not in st.session_state:
    st.session_state.work_order_triggered = False

if "step_count" not in st.session_state:
    st.session_state.step_count = 0

# 4. Sidebar Controls (Static Layout)
st.sidebar.markdown("<h2 style='color: #00f2fe; font-weight:800; font-size:1.4rem;'>⚙️ TWIN CONTROL HUB</h2>", unsafe_allow_html=True)

selected_asset = st.sidebar.selectbox(
    "Target Industrial Asset",
    ["Gas Turbine #1 (Primary)", "CNC Milling Center #2", "Hydraulic Pump Unit #3"]
)

stream_active = st.sidebar.toggle("Stream Live Telemetry", value=True)
sim_speed = st.sidebar.slider("Update Interval (Seconds)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color: #a855f7; font-size:1rem; font-weight:700;'>⚡ FAULT INJECTION STUDIO</h4>", unsafe_allow_html=True)

col_f1, col_f2 = st.sidebar.columns(2)
if col_f1.button("🟢 Normal Mode", use_container_width=True):
    st.session_state.fault_mode = "NORMAL"
if col_f2.button("🔥 Thermal Spike", use_container_width=True):
    st.session_state.fault_mode = "THERMAL"

col_f3, col_f4 = st.sidebar.columns(2)
if col_f3.button("💥 Bearing Fault", use_container_width=True):
    st.session_state.fault_mode = "VIBRATION"
if col_f4.button("🚨 Cascade Failure", use_container_width=True):
    st.session_state.fault_mode = "CASCADE"

load_spike = st.sidebar.slider("Apply Dynamic Load Offset (%)", 0, 50, 0, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color: #ff9f43; font-size:1rem; font-weight:700;'>🛡️ SAFETY LIMITS</h4>", unsafe_allow_html=True)
thresh_vib = st.sidebar.slider("Vibration Threshold (mm/s)", 1.0, 12.0, 6.5, 0.1)
thresh_temp = st.sidebar.slider("Temperature Threshold (°C)", 40, 120, 82, 1)
thresh_curr = st.sidebar.slider("Current Limit (A)", 5.0, 50.0, 32.0, 0.5)

# 5. Helper Plotly Renderers
def render_gauge(value, min_v, max_v, title, unit, limit, color_hex):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"<b>{title}</b><br><span style='font-size:0.7em;color:#94a3b8;'>Limit: {limit} {unit}</span>", 'font': {'size': 12, 'color': "#ffffff"}},
        number={'suffix': f" {unit}", 'font': {'size': 20, 'color': color_hex, 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [min_v, max_v], 'tickcolor': "#475569", 'tickwidth': 1},
            'bar': {'color': color_hex},
            'bgcolor': "rgba(15, 23, 42, 0.6)",
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'threshold': {
                'line': {'color': "#ff1744", 'width': 3},
                'thickness': 0.8,
                'value': limit
            }
        }
    ))
    fig.update_layout(
        height=170,
        margin=dict(l=15, r=15, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig

def render_line_chart(df, y_col, title, color_hex, threshold_val=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df[y_col],
        mode="lines+markers",
        line=dict(color=color_hex, width=2.5),
        marker=dict(size=4, color=color_hex),
        name=y_col
    ))
    
    if threshold_val is not None:
        fig.add_hline(
            y=threshold_val,
            line_dash="dash",
            line_color="#ff1744",
            line_width=1.5,
            annotation_text="Limit",
            annotation_position="top right"
        )

    fig.update_layout(
        title={'text': title, 'font': {'size': 13, 'color': '#e2e8f0'}},
        paper_bgcolor="rgba(15, 23, 42, 0.5)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        font=dict(color="#94a3b8", size=10),
        height=210,
        margin=dict(l=25, r=20, t=35, b=25),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
    )
    return fig

def generate_sensor_sample(current_time):
    mode = st.session_state.fault_mode
    load_multiplier = 1.0 + (load_spike / 100.0)

    if mode == "NORMAL":
        vib = round(np.random.normal(5.1, 0.4) * load_multiplier, 2)
        temp = round(np.random.normal(71.0, 1.2) * load_multiplier, 1)
        curr = round(np.random.normal(23.5, 1.0) * load_multiplier, 1)
    elif mode == "THERMAL":
        vib = round(np.random.normal(5.6, 0.5) * load_multiplier, 2)
        temp = round(np.random.normal(89.5, 3.2) * load_multiplier, 1)
        curr = round(np.random.normal(29.0, 1.5) * load_multiplier, 1)
    elif mode == "VIBRATION":
        vib = round(np.random.normal(8.7, 1.1) * load_multiplier, 2)
        temp = round(np.random.normal(76.0, 1.8) * load_multiplier, 1)
        curr = round(np.random.normal(27.0, 1.2) * load_multiplier, 1)
    elif mode == "CASCADE":
        vib = round(np.random.normal(9.8, 1.4) * load_multiplier, 2)
        temp = round(np.random.normal(96.0, 4.0) * load_multiplier, 1)
        curr = round(np.random.normal(38.0, 2.5) * load_multiplier, 1)

    power = round((curr * 400 * 1.732 * 0.88) / 1000, 1)

    r_vib = vib / thresh_vib
    r_temp = temp / thresh_temp
    r_curr = curr / thresh_curr
    max_r = max(r_vib, r_temp, r_curr)
    
    anomaly_score = min(round((max_r ** 2) * 35, 1), 99.9)
    status = "CRITICAL" if max_r >= 1.0 else ("WARNING" if max_r >= 0.85 else "NORMAL")

    return {
        "Timestamp": current_time,
        "Asset": selected_asset,
        "Vibration": vib,
        "Temperature": temp,
        "Current": curr,
        "Power": power,
        "Anomaly_Score": anomaly_score,
        "Status": status
    }

# 6. Streamlit Isolated Fragment Engine (Eliminates Full Page Blinking)
@st.fragment(run_every=sim_speed if stream_active else None)
def render_live_telemetry():
    current_time = datetime.now().astimezone().strftime("%H:%M:%S")

    # Header Render
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div style="display:flex; align-items:center; gap:12px;">
                <h1 style="margin:0; font-size:1.6rem; font-weight:800; letter-spacing:1px; color:#ffffff;">
                    INDUSTRIAL<span style="color:#00f2fe;">TWIN.AI</span>
                </h1>
                <span style="background:rgba(0,242,254,0.15); color:#00f2fe; border:1px solid rgba(0,242,254,0.4); padding:2px 10px; border-radius:12px; font-size:0.7rem; font-weight:700;">PRO ENGINE v3.4</span>
            </div>
            <p style="margin:4px 0 0 0; font-size:0.8rem; color:#94a3b8;">Autonomous Predictive AI Telemetry & Physics-Based Twin Engine</p>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.75rem; color:#64748b; font-weight:600;">SYSTEM CLOCK (LOCAL)</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.2rem; font-weight:800; color:#00f2fe;">
                {current_time}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data Update Step
    sample = generate_sensor_sample(current_time)
    st.session_state.step_count += 1
    step_idx = st.session_state.step_count

    new_row = pd.DataFrame([sample])
    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True).tail(30)
    df_curr = st.session_state.data

    is_critical = sample["Status"] == "CRITICAL"
    health_idx = max(round(100.0 - sample["Anomaly_Score"], 1), 0.1)
    rul_hours = int((health_idx / 100.0) * 1250)

    # Alert Section
    if is_critical:
        breaches = []
        if sample["Vibration"] > thresh_vib: breaches.append(f"Vibration ({sample['Vibration']} mm/s)")
        if sample["Temperature"] > thresh_temp: breaches.append(f"Temperature ({sample['Temperature']} °C)")
        if sample["Current"] > thresh_curr: breaches.append(f"Current Draw ({sample['Current']} A)")
        
        st.markdown(f"""
            <div class="ai-box-danger">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="badge-danger">🚨 CRITICAL BREACH DETECTED</span>
                        <h3 style="margin:6px 0 2px 0; color:#ff1744; font-size:1.1rem; font-weight:800;">
                            Safety Envelope Exceeded on {selected_asset}
                        </h3>
                        <p style="margin:0; font-size:0.85rem; color:#fca5a5;">
                            Parameter Limit Violations: <b>{', '.join(breaches)}</b>. Automated safety interlock advisory issued.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="ai-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="badge-ok">✅ SYSTEM OPTIMAL</span>
                        <span style="font-size:0.9rem; font-weight:700; color:#e2e8f0; margin-left:8px;">
                            {selected_asset} operating in Nominal Envelope
                        </span>
                        <p style="margin:4px 0 0 0; font-size:0.8rem; color:#94a3b8;">
                            Active Mode: <b>{st.session_state.fault_mode}</b> | All dynamic telemetry channels stabilized.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Metric Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">OVERALL HEALTH</div>
            <div class="metric-value" style="color:{'#ff1744' if health_idx < 60 else '#00e676'};">{health_idx}%</div>
            <div class="metric-sub" style="color:#94a3b8;">Status: {sample['Status']}</div>
        </div>
    """, unsafe_allow_html=True)

    k2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ANOMALY RISK</div>
            <div class="metric-value" style="color:{'#ff1744' if sample['Anomaly_Score'] > 50 else '#00f2fe'};">{sample['Anomaly_Score']}%</div>
            <div class="metric-sub" style="color:#94a3b8;">Weighted Risk Model</div>
        </div>
    """, unsafe_allow_html=True)

    k3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ESTIMATED RUL</div>
            <div class="metric-value" style="color:#a855f7;">{rul_hours} <span style="font-size:1rem;">hrs</span></div>
            <div class="metric-sub" style="color:#94a3b8;">~{int(rul_hours/24)} Days Remaining</div>
        </div>
    """, unsafe_allow_html=True)

    k4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">POWER DRAW</div>
            <div class="metric-value" style="color:#ff9f43;">{sample['Power']} <span style="font-size:1rem;">kW</span></div>
            <div class="metric-sub" style="color:#94a3b8;">Draw @ 400V 3-Phase</div>
        </div>
    """, unsafe_allow_html=True)

    k5.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">INJECTED FAULT</div>
            <div class="metric-value" style="font-size:1.2rem; color:#f43f5e; padding-top:6px;">{st.session_state.fault_mode}</div>
            <div class="metric-sub" style="color:#94a3b8;">Offset: +{load_spike}% Load</div>
        </div>
    """, unsafe_allow_html=True)

    # Tabs Layout
    tab_telemetry, tab_ai_engine, tab_audit = st.tabs([
        "📡 Live Telemetry Twin", 
        "🤖 Predictive AI & Cost Diagnostics", 
        "⚙️ Asset Audit Logs"
    ])

    with tab_telemetry:
        g1, g2, g3, g4 = st.columns(4)
        g1.plotly_chart(render_gauge(sample["Vibration"], 0, 12, "VIBRATION", "mm/s", thresh_vib, "#00f2fe"), use_container_width=True, key=f"g1_{step_idx}")
        g2.plotly_chart(render_gauge(sample["Temperature"], 30, 120, "TEMPERATURE", "°C", thresh_temp, "#ff9f43"), use_container_width=True, key=f"g2_{step_idx}")
        g3.plotly_chart(render_gauge(sample["Current"], 0, 50, "CURRENT DRAW", "A", thresh_curr, "#a855f7"), use_container_width=True, key=f"g3_{step_idx}")
        g4.plotly_chart(render_gauge(sample["Power"], 0, 30, "ACTIVE POWER", "kW", 24.0, "#00e676"), use_container_width=True, key=f"g4_{step_idx}")

        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(render_line_chart(df_curr, "Vibration", "Vibration Signature (mm/s)", "#00f2fe", thresh_vib), use_container_width=True, key=f"c1_{step_idx}")
        c2.plotly_chart(render_line_chart(df_curr, "Temperature", "Thermal Profile (°C)", "#ff9f43", thresh_temp), use_container_width=True, key=f"c2_{step_idx}")
        c3.plotly_chart(render_line_chart(df_curr, "Current", "Current Consumption (A)", "#a855f7", thresh_curr), use_container_width=True, key=f"c3_{step_idx}")

    with tab_ai_engine:
        ai_col_left, ai_col_right = st.columns([1.6, 1])
        with ai_col_left:
            fig_ai = go.Figure()
            fig_ai.add_trace(go.Scatter(
                x=df_curr["Timestamp"],
                y=df_curr["Anomaly_Score"],
                fill='tozeroy',
                fillcolor='rgba(255, 23, 68, 0.2)' if is_critical else 'rgba(0, 242, 254, 0.15)',
                line=dict(color="#ff1744" if is_critical else "#00f2fe", width=3),
                name="Risk Score %"
            ))
            fig_ai.update_layout(
                title={'text': "<b>XGBoost Multi-Sensor Anomaly Probability Trend</b>", 'font': {'size': 14, 'color': '#ffffff'}},
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font=dict(color="#94a3b8"),
                height=260,
                margin=dict(l=20, r=20, t=35, b=20),
                yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_ai, use_container_width=True, key=f"ai_chart_{step_idx}")

            bearing_health = max(round(100 - (sample["Vibration"] / thresh_vib) * 45, 1), 5.0)
            stator_health = max(round(100 - (sample["Current"] / thresh_curr) * 40, 1), 5.0)
            thermal_barrier = max(round(100 - (sample["Temperature"] / thresh_temp) * 42, 1), 5.0)

            st.markdown(f"""
                <div class="metric-card" style="margin-top:10px;">
                    <div style="font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:12px;">
                        SUBSYSTEM HEALTH DEGRADATION BREAKDOWN
                    </div>
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#cbd5e1; margin-bottom:4px;">
                            <span>Rotor Bearing Assembly</span>
                            <span style="font-family:'JetBrains Mono'; font-weight:700; color:{'#ff1744' if bearing_health < 50 else '#00e676'};">{bearing_health}%</span>
                        </div>
                        <div style="background:#1e293b; border-radius:6px; height:8px; overflow:hidden;">
                            <div style="background:{'#ff1744' if bearing_health < 50 else '#00e676'}; width:{bearing_health}%; height:100%;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#cbd5e1; margin-bottom:4px;">
                            <span>Stator Winding Insulation</span>
                            <span style="font-family:'JetBrains Mono'; font-weight:700; color:{'#ff1744' if stator_health < 50 else '#00e676'};">{stator_health}%</span>
                        </div>
                        <div style="background:#1e293b; border-radius:6px; height:8px; overflow:hidden;">
                            <div style="background:{'#ff1744' if stator_health < 50 else '#00e676'}; width:{stator_health}%; height:100%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#cbd5e1; margin-bottom:4px;">
                            <span>Turbine Thermal Barrier</span>
                            <span style="font-family:'JetBrains Mono'; font-weight:700; color:{'#ff1744' if thermal_barrier < 50 else '#00e676'};">{thermal_barrier}%</span>
                        </div>
                        <div style="background:#1e293b; border-radius:6px; height:8px; overflow:hidden;">
                            <div style="background:{'#ff1744' if thermal_barrier < 50 else '#00e676'}; width:{thermal_barrier}%; height:100%;"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with ai_col_right:
            if is_critical:
                st.markdown(f"""
                    <div class="ai-box-danger">
                        <h4 style="margin:0 0 8px 0; color:#ff1744; font-size:0.95rem;">🤖 AI DIAGNOSTIC COPILOT: ANOMALY</h4>
                        <p style="font-size:0.8rem; color:#e2e8f0; line-height:1.4;">
                            <b>Primary Driver:</b> Fault mode <code>{st.session_state.fault_mode}</code> triggered excessive telemetry divergence.
                        </p>
                        <div style="background:rgba(0,0,0,0.4); padding:10px; border-radius:8px; font-size:0.75rem; font-family:'JetBrains Mono'; color:#fca5a5; margin-top:8px;">
                            💡 Prescriptive Action: Reduce operating load by 25% & dispatch Maintenance Crew A to inspect asset immediately.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="ai-box">
                        <h4 style="margin:0 0 8px 0; color:#00f2fe; font-size:0.95rem;">🤖 AI DIAGNOSTIC COPILOT: NOMINAL</h4>
                        <p style="font-size:0.8rem; color:#cbd5e1; line-height:1.4;">
                            <b>Harmonic Analysis:</b> Zero mechanical unbalance or thermal runaway signatures detected.
                        </p>
                        <div style="background:rgba(0,0,0,0.4); padding:10px; border-radius:8px; font-size:0.75rem; font-family:'JetBrains Mono'; color:#67e8f9; margin-top:8px;">
                            💡 Prescriptive Action: Continue standard operation. Next planned downtime cycle in 42 days.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            estimated_downtime_cost = round((sample["Anomaly_Score"] / 100.0) * 14500, 2)
            st.markdown(f"""
                <div class="metric-card" style="margin-top:12px;">
                    <div class="metric-label">ESTIMATED DOWNTIME COST RISK</div>
                    <div class="metric-value" style="color:#ff9f43;">${estimated_downtime_cost:,.2f} <span style="font-size:0.8rem; color:#94a3b8;">/ hr</span></div>
                    <p style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Based on current production rate and risk rating.</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚨 TRIGGER WORK ORDER TICKET", type="primary", use_container_width=True, key=f"btn_wo_{step_idx}"):
                st.session_state.work_order_triggered = True

            if st.session_state.work_order_triggered:
                st.success("Work Order Ticket successfully dispatched to Engineering Maintenance Team!")

    with tab_audit:
        st.markdown("<h4 style='color:#e2e8f0; font-size:1rem; margin-bottom:12px;'>Real-Time Telemetry Data Stream Audit</h4>", unsafe_allow_html=True)
        st.dataframe(
            df_curr.sort_values(by="Timestamp", ascending=False),
            use_container_width=True,
            column_config={
                "Status": st.column_config.TextColumn("Status"),
                "Anomaly_Score": st.column_config.NumberColumn("Anomaly Risk %", format="%.1f%%")
            }
        )

        csv_data = df_curr.to_csv(index=False).encode('utf-8')
        export_col1, export_col2 = st.columns([1, 4])
        export_col1.download_button(
            label="📥 Export Telemetry CSV",
            data=csv_data,
            file_name=f"telemetry_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"btn_dl_{step_idx}"
        )

# Execute Fragment Container
render_live_telemetry()
