import streamlit as st
import json
import os
import sys
import pandas as pd
import altair as alt
import uuid
from datetime import datetime

st.set_page_config(
    page_title="RoadGuardian Cockpit & DMS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Cockpit styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }
    .alert-card {
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.9em;
        font-family: monospace;
        font-weight: bold;
    }
    .cockpit-title {
        font-size: 24px;
        font-weight: 800;
        color: #f3f4f6;
        margin-bottom: 2px;
    }
    .cockpit-sub {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="cockpit-title">🚗 ROADGUARDIAN COCKPIT</div>', unsafe_allow_html=True)
st.markdown('<div class="cockpit-sub">Vehicle Safety Intelligence & Event Data Recorder</div>', unsafe_allow_html=True)

# Configuration options in sidebar
st.sidebar.header("🔧 DMS & Cockpit Config")
input_video_path = st.sidebar.text_input("Input Video Path", "input/driver.mp4")
processed_video_path = st.sidebar.text_input("Processed Video Path", "output/processed_driver.mp4")
telemetry_path = st.sidebar.text_input("Telemetry JSON Path", "output/telemetry.json")

# Force Processing Trigger
if st.sidebar.button("⚙️ Re-process Video"):
    with st.spinner("Processing video frame-by-frame... (using CPU delegate)"):
        import subprocess
        result = subprocess.run([
            sys.executable, "process_video.py", 
            "--input", input_video_path,
            "--output", processed_video_path,
            "--telemetry", telemetry_path
        ], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Video processed successfully!")
        else:
            st.sidebar.error(f"Error processing: {result.stderr}")

# Check files
has_video = os.path.exists(processed_video_path)
has_telemetry = os.path.exists(telemetry_path)

if not has_video or not has_telemetry:
    st.info("💡 Please process the driver video first using the sidebar configuration button or the command: `python process_video.py`.")

# Load telemetry
telemetry = None
if has_telemetry:
    try:
        with open(telemetry_path, "r") as f:
            telemetry = json.load(f)
    except Exception as e:
        st.error(f"Error loading telemetry: {e}")

# Main cockpit grids
if telemetry and "records" in telemetry:
    records = telemetry["records"]
    df = pd.DataFrame(records)
    
    # 1. Timeline Explorer Slider (Syncs the entire cockpit dashboard!)
    st.markdown("### ⏱️ Cockpit Timeline Explorer")
    max_time = float(df["timestamp_sec"].max())
    selected_time = st.slider("Select Time (seconds)", 0.0, max_time, 0.0, step=0.2)
    
    # Find closest telemetry record
    idx = (df["timestamp_sec"] - selected_time).abs().idxmin()
    record = records[idx]

    # Status Indicators Header — safe .get() for backwards compatibility
    status          = record.get("event_status", "NORMAL")
    event_type      = record.get("event_type", "NORMAL")
    type_conf       = record.get("type_confidence", 0.0)
    status_conf     = record.get("status_confidence", 0.0)
    speed_val       = record.get("speed", 0.0)
    braking_val     = record.get("braking", 0.0)
    distance_val    = record.get("distance", 0.0)
    hazard_val      = record.get("hazard", 0)
    visibility_val  = record.get("visibility", 1.0)

    if status == "CRITICAL":
        st.error(f"🔴 CRITICAL EVENT DETECTED (Type: {event_type} - Confidence: {type_conf:.1%})")
    elif status in ("WARNING", "RISK"):
        st.warning(f"🟡 RISK DEVELOPING DETECTED (Confidence: {status_conf:.1%})")
    else:
        st.success("🟢 NORMAL DRIVE ACTIVE")

    # Dual Column Layout for Cameras
    col_cam1, col_cam2 = st.columns(2)
    with col_cam1:
        st.markdown("#### 📹 Internal Camera (DMS)")
        if has_video:
            st.video(processed_video_path)
        else:
            st.warning("Processed driver video not found.")
            
    with col_cam2:
        st.markdown("#### 🛣️ External Camera (Road Perception - Driver's POV)")
        st.info("Road Perception Model Interface (Car, Bus, Pedestrian Detection placeholder)")
        st.markdown(f"""
        <div style="background-color: #111827; padding: 25px; border-radius: 8px; border: 1px solid {'#ef4444' if hazard_val else '#374151'}; text-align: center;">
            <div style="font-size: 11px; color: #9ca3af; letter-spacing: 1px;">EXTERNAL LIVE TARGETS</div>
            <div style="font-size: 32px; font-weight: bold; margin-top: 10px; color: {'#ef4444' if hazard_val else '#10b981'};">
                {"⚠️ HAZARD AHEAD" if hazard_val else "🛣️ ROAD CLEAR"}
            </div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 5px;">
                Nearest Object Distance: <strong>{distance_val:.1f} m</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3-Column Panel Layout for Cockpit Stats
    st.markdown("---")
    panel_v, panel_d, panel_e = st.columns(3)

    with panel_v:
        st.markdown("### 🚘 Vehicle State")
        st.metric("Current Speed", f"{speed_val:.1f} km/h")
        st.metric("Braking Level", f"{braking_val:.0%}")
        st.metric("Nearest Vehicle Distance", f"{distance_val:.1f} m")

    with panel_d:
        st.markdown("### 👤 Driver State (DMS)")
        st.metric("Driver Distraction Score", f"{record['distraction_score']} / 100")
        st.metric("Driver Drowsiness Score", f"{record['drowsiness_score']} / 100")
        st.metric("Attention Index Score", f"{record['attention_score']} / 100")

    with panel_e:
        st.markdown("### 🌍 Surroundings & Environment")
        st.metric("Atmospheric Visibility", f"{visibility_val:.0%}")
        st.metric("Road Hazard Detection", "DETECTED 🔴" if hazard_val else "CLEAR 🟢")
        st.metric("Head Pose Direction", record["head_pose_state"])

    # Event logs and Alert Event lists
    st.markdown("---")
    col_alerts, col_chart = st.columns([1, 2])
    
    # Compile alert event logs dynamically
    alert_logs = []
    active_incidents = {}
    for r in records:
        timestamp = f"{int(r['timestamp_sec'] // 60)}:{int(r['timestamp_sec'] % 60):02d}"
        for alert_label in r["alerts"]:
            clean_label = alert_label.strip()
            if clean_label not in active_incidents:
                active_incidents[clean_label] = timestamp
                alert_logs.append(f"[{timestamp}] ⚠️ {clean_label} STARTED")
        
        ended_alerts = []
        for active_alert in active_incidents:
            match_found = False
            for current_alert in r["alerts"]:
                if current_alert.strip() == active_alert:
                    match_found = True
                    break
            if not match_found:
                ended_alerts.append(active_alert)
        
        for alert_to_remove in ended_alerts:
            start_time = active_incidents.pop(alert_to_remove)
            alert_logs.append(f"[{timestamp}] ✅ {alert_to_remove} RESOLVED (Active since {start_time})")

    with col_alerts:
        st.subheader("🚨 Safety Alerts History")
        if alert_logs:
            for log in alert_logs[-8:]:
                if "RESOLVED" in log:
                    card_style = "background-color: #064e3b; color: #a7f3d0; border-left: 5px solid #10b981;"
                else:
                    card_style = "background-color: #7f1d1d; color: #fca5a5; border-left: 5px solid #ef4444;"
                st.markdown(f"<div class='alert-card' style='{card_style}'>{log}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ No critical driver infractions detected.")

    with col_chart:
        st.subheader("📈 Telemetry Waveforms")
        chart_df = df[["timestamp_sec", "attention_score", "drowsiness_score", "distraction_score"]]
        chart_df.columns = ["Time (s)", "Attention Score", "Drowsiness Score", "Distraction Score"]
        melted_df = chart_df.melt("Time (s)", var_name="Metric", value_name="Score")
        
        line_chart = alt.Chart(melted_df).mark_line().encode(
            x='Time (s):Q',
            y='Score:Q',
            color='Metric:N'
        ).properties(height=230)
        st.altair_chart(line_chart, use_container_width=True)

    # Black Box / Incident Data Recorder (EDR) Section
    st.markdown("---")
    st.subheader("📦 RoadGuardian Event Data Recorder (Black Box)")
    
    blackbox_csv_path = "output/roadguardian_blackbox_event.csv"
    incident_summary_path = "output/roadguardian_incident_summary.json"
    
    if os.path.exists(blackbox_csv_path) and os.path.exists(incident_summary_path):
        st.error("🔒 VEHICLE BLACK BOX ACTIVATED — EDR INCIDENT DUMP PRESERVED")
        
        # Load incident summary
        with open(incident_summary_path, "r") as f:
            incident_data = json.load(f)
            
        # Display incident metadata
        st.markdown(f"""
        * **Event ID:** `{incident_data['event_id']}`
        * **Event Status / Class:** `{incident_data['event_status']}` (Confidence: `{incident_data['status_confidence']:.1%}`)
        * **Event Type / Incident:** `{incident_data['event_type']}` (Confidence: `{incident_data['type_confidence']:.1%}`)
        * **Trigger Values:** Speed: `{incident_data['trigger_speed_kmh']:.1f} km/h` | Deceleration: `{incident_data['trigger_braking_level']:.0%}` | Distraction: `{incident_data['trigger_distraction']:.0%}` | Drowsiness: `{incident_data['trigger_drowsiness']:.0%}`
        """)
        
        # Load EDR table
        blackbox_df = pd.read_csv(blackbox_csv_path)
        
        st.dataframe(blackbox_df[[
            "time", "speed", "acceleration", "braking", "nearest_vehicle_distance",
            "driver_distraction", "driver_drowsiness", "event_status", "event_type"
        ]])
        
        # CSV download button
        with open(blackbox_csv_path, "r") as f:
            st.download_button(
                label="📥 Download Event Data Recorder (EDR) CSV",
                data=f.read(),
                file_name="roadguardian_blackbox_event.csv",
                mime="text/csv"
            )
    else:
        st.info("ℹ️ Black box is continuously maintaining a rolling 15-second pre-event buffer. A critical event (like sudden deceleration / crash) will lock the preceding history and record the aftermath.")

else:
    st.info("Waiting for telemetry data...")
