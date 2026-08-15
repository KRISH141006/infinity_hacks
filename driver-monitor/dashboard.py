import streamlit as st
import json
import os
import sys
import pandas as pd
import altair as alt
import subprocess
from datetime import datetime

st.set_page_config(
    page_title="RoadGuardian Cockpit - DMS & ADAS Intelligence",
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

st.markdown('<div class="cockpit-title">🚗 ROADGUARDIAN UNIFIED COCKPIT</div>', unsafe_allow_html=True)
st.markdown('<div class="cockpit-sub">Driver Monitoring System (DMS) & Advanced Road Perception (ADAS)</div>', unsafe_allow_html=True)

# Configuration options in sidebar
st.sidebar.header("🔧 DMS Config (Internal Camera)")
# Pick sensible default if tired_driver_processed.mp4 exists
default_driver_in = "input/tired_driver.webm" if os.path.exists("input/tired_driver.webm") else "input/driver.mp4"
default_driver_out = "output/tired_driver_processed.mp4" if os.path.exists("output/tired_driver_processed.mp4") else "output/processed_driver.mp4"

input_video_path = st.sidebar.text_input("Internal Input Video", default_driver_in)
processed_video_path = st.sidebar.text_input("Internal Processed Video", default_driver_out)
telemetry_path = st.sidebar.text_input("Driver Telemetry JSON", "output/telemetry.json")

if st.sidebar.button("👤 Process Driver DMS Video"):
    with st.spinner("Processing driver monitoring video..."):
        result = subprocess.run([
            sys.executable, "process_video.py", 
            "--input", input_video_path,
            "--output", processed_video_path,
            "--telemetry", telemetry_path
        ], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Driver DMS video processed successfully!")
            st.rerun()
        else:
            st.sidebar.error(f"Error processing driver video: {result.stderr}")

st.sidebar.markdown("---")
st.sidebar.header("🛣️ ADAS Config (External Camera)")
input_road_path = st.sidebar.text_input("External Road Video", "input/external_road.mp4")
processed_road_path = st.sidebar.text_input("External Processed Video", "output/processed_road.mp4")
road_telemetry_path = st.sidebar.text_input("Road Telemetry JSON", "output/road_telemetry.json")

if st.sidebar.button("🛣️ Process Road ADAS Video"):
    with st.spinner("Processing external camera labeled video & tracking..."):
        result = subprocess.run([
            sys.executable, "process_road_video.py",
            "--input", input_road_path,
            "--output", processed_road_path,
            "--telemetry", road_telemetry_path
        ], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("External road video labeled & processed successfully!")
            st.rerun()
        else:
            st.sidebar.error(f"Error processing road video: {result.stderr}")

# Check files availability
has_driver_video = os.path.exists(processed_video_path)
has_driver_telemetry = os.path.exists(telemetry_path)
has_road_video = os.path.exists(processed_road_path)
has_road_telemetry = os.path.exists(road_telemetry_path)

# Load Driver Telemetry
driver_telemetry = None
if has_driver_telemetry:
    try:
        with open(telemetry_path, "r") as f:
            driver_telemetry = json.load(f)
    except Exception as e:
        st.error(f"Error loading driver telemetry: {e}")

# Load Road Telemetry
road_telemetry = None
if has_road_telemetry:
    try:
        with open(road_telemetry_path, "r") as f:
            road_telemetry = json.load(f)
    except Exception as e:
        st.error(f"Error loading road telemetry: {e}")

if not has_driver_video and not has_road_video:
    st.info("💡 Please process the driver video or road video using the sidebar buttons.")

# Determine Timeline Slider parameters
driver_records = driver_telemetry.get("records", []) if driver_telemetry else []
road_records = road_telemetry.get("records", []) if road_telemetry else []

df_driver = pd.DataFrame(driver_records) if driver_records else None
df_road = pd.DataFrame(road_records) if road_records else None

max_time = 0.0
if df_driver is not None and not df_driver.empty:
    max_time = max(max_time, float(df_driver["timestamp_sec"].max()))
if df_road is not None and not df_road.empty:
    max_time = max(max_time, float(df_road["timestamp_sec"].max()))

if max_time > 0.0:
    st.markdown("### ⏱️ Cockpit Timeline Explorer")
    selected_time = st.slider("Timeline Synchronization (seconds)", 0.0, max_time, 0.0, step=0.1)
    
    # Retrieve current driver record
    curr_driver = {}
    if df_driver is not None and not df_driver.empty:
        idx_d = (df_driver["timestamp_sec"] - selected_time).abs().idxmin()
        curr_driver = driver_records[idx_d]
        
    # Retrieve current road record
    curr_road = {}
    if df_road is not None and not df_road.empty:
        idx_r = (df_road["timestamp_sec"] - selected_time).abs().idxmin()
        curr_road = road_records[idx_r]

    # Global status banner
    status = curr_driver.get("event_status", "NORMAL")
    collision_risk = curr_road.get("collision_risk", "Low")
    road_hazard = curr_road.get("road_hazard", 0) or curr_driver.get("hazard", 0)

    if status == "CRITICAL" or collision_risk == "High":
        st.error(f"🔴 CRITICAL SAFETY ALERT (Collision Risk: {collision_risk.upper()} | Driver Status: {status})")
    elif status in ("WARNING", "RISK") or collision_risk == "Medium" or road_hazard:
        st.warning(f"🟡 ADAS WARNING (Hazard: {'YES' if road_hazard else 'NO'} | Risk: {collision_risk.upper()} | Driver Status: {status})")
    else:
        st.success("🟢 NORMAL SECURE DRIVE ACTIVE")

    # Dual Column Layout for Cameras
    col_cam1, col_cam2 = st.columns(2)
    with col_cam1:
        st.markdown("#### 📹 Internal Camera (Driver Monitoring System)")
        if has_driver_video:
            st.video(processed_video_path)
        else:
            st.warning("Processed driver video not found. Run processing in the sidebar.")
            
    with col_cam2:
        st.markdown("#### 🛣️ External Camera (Road Perception - Driver's POV)")
        if has_road_video:
            st.video(processed_road_path)
        else:
            st.info("Processed labeled road video will appear here once processed.")
            st.markdown(f"""
            <div style="background-color: #111827; padding: 25px; border-radius: 8px; border: 1px solid {'#ef4444' if road_hazard else '#374151'}; text-align: center;">
                <div style="font-size: 11px; color: #9ca3af; letter-spacing: 1px;">EXTERNAL LIVE TARGETS</div>
                <div style="font-size: 32px; font-weight: bold; margin-top: 10px; color: {'#ef4444' if road_hazard else '#10b981'};">
                    {"⚠️ HAZARD AHEAD" if road_hazard else "🛣️ ROAD CLEAR"}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 3-Column Panel Layout for Cockpit Stats
    st.markdown("---")
    panel_v, panel_d, panel_e = st.columns(3)

    speed_val = curr_driver.get("speed", 45.0)
    braking_val = curr_driver.get("braking", 0.0)
    nearest_dist = curr_road.get("nearest_vehicle_distance", curr_driver.get("distance", 0.0))
    lane_stat = curr_road.get("lane_status", "Lane Center")

    with panel_v:
        st.markdown("### 🚘 Vehicle & ADAS State")
        st.metric("Estimated Speed", f"{speed_val:.1f} km/h")
        st.metric("Lane Position", lane_stat)
        st.metric("Nearest Target Distance", f"{nearest_dist:.1f} m" if nearest_dist > 0 else "Clear")

    with panel_d:
        st.markdown("### 👤 Driver State (DMS)")
        st.metric("Driver Distraction Score", f"{curr_driver.get('distraction_score', 0)} / 100")
        st.metric("Driver Drowsiness Score", f"{curr_driver.get('drowsiness_score', 0)} / 100")
        st.metric("Attention Index Score", f"{curr_driver.get('attention_score', 100)} / 100")

    with panel_e:
        st.markdown("### 🌍 Road Perception & Targets")
        veh_count = curr_road.get("vehicle_count", 0)
        ped_count = curr_road.get("pedestrian_count", 0)
        st.metric("Vehicles Ahead", f"{veh_count} detected")
        st.metric("Pedestrians in View", f"{ped_count} detected")
        st.metric("Collision Risk", collision_risk.upper())

    # Alerts & Telemetry Waveforms
    st.markdown("---")
    col_alerts, col_chart = st.columns([1, 2])

    alert_logs = []
    if df_driver is not None and not df_driver.empty and "alerts" in df_driver.columns:
        active_incidents = {}
        for r in driver_records:
            t_str = f"{int(r['timestamp_sec'] // 60)}:{int(r['timestamp_sec'] % 60):02d}"
            for alert_label in r.get("alerts", []):
                clean_label = alert_label.strip()
                if clean_label not in active_incidents:
                    active_incidents[clean_label] = t_str
                    alert_logs.append(f"[{t_str}] ⚠️ {clean_label} STARTED")
            
            ended = [k for k in active_incidents if not any(k == a.strip() for a in r.get("alerts", []))]
            for k in ended:
                start_t = active_incidents.pop(k)
                alert_logs.append(f"[{t_str}] ✅ {k} RESOLVED (Active since {start_t})")

    # Also include Road Events if present
    road_events_path = "output/road_events.json"
    if os.path.exists(road_events_path):
        try:
            with open(road_events_path, "r") as rf:
                r_events = json.load(rf)
                for rev in r_events:
                    rt_str = f"{int(rev['timestamp'] // 60)}:{int(rev['timestamp'] % 60):02d}"
                    alert_logs.append(f"[{rt_str}] 🛣️ ADAS EVENT: {rev['event']} (Value: {rev['value']})")
        except Exception:
            pass

    with col_alerts:
        st.subheader("🚨 Safety Alerts History")
        if alert_logs:
            for log in alert_logs[-8:]:
                if "RESOLVED" in log:
                    card_style = "background-color: #064e3b; color: #a7f3d0; border-left: 5px solid #10b981;"
                elif "ADAS" in log:
                    card_style = "background-color: #1e3a8a; color: #bfdbfe; border-left: 5px solid #3b82f6;"
                else:
                    card_style = "background-color: #7f1d1d; color: #fca5a5; border-left: 5px solid #ef4444;"
                st.markdown(f"<div class='alert-card' style='{card_style}'>{log}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ No critical infractions detected.")

    with col_chart:
        st.subheader("📈 Telemetry & Perception Waveforms")
        if df_driver is not None and not df_driver.empty:
            chart_df = df_driver[["timestamp_sec", "attention_score", "drowsiness_score", "distraction_score"]].copy()
            chart_df.columns = ["Time (s)", "Attention Score", "Drowsiness Score", "Distraction Score"]
            melted_df = chart_df.melt("Time (s)", var_name="Metric", value_name="Score")
            
            line_chart = alt.Chart(melted_df).mark_line().encode(
                x='Time (s):Q',
                y='Score:Q',
                color='Metric:N'
            ).properties(height=230)
            st.altair_chart(line_chart, use_container_width=True)
        elif df_road is not None and not df_road.empty:
            chart_df = df_road[["timestamp_sec", "vehicle_count", "nearest_vehicle_distance"]].copy()
            chart_df.columns = ["Time (s)", "Vehicle Count", "Nearest Distance (m)"]
            melted_df = chart_df.melt("Time (s)", var_name="Metric", value_name="Value")
            line_chart = alt.Chart(melted_df).mark_line().encode(
                x='Time (s):Q',
                y='Value:Q',
                color='Metric:N'
            ).properties(height=230)
            st.altair_chart(line_chart, use_container_width=True)

    # Black Box & Witness Discovery Section
    st.markdown("---")
    st.subheader("📦 RoadGuardian Black Box & Geofence Witness Discovery")
    
    col_bb, col_wit = st.columns(2)
    
    with col_bb:
        st.markdown("#### 🔒 Incident Data Recorder (EDR)")
        blackbox_csv_path = "output/roadguardian_blackbox_event.csv"
        incident_summary_path = "output/roadguardian_incident_summary.json"
        
        if os.path.exists(blackbox_csv_path) and os.path.exists(incident_summary_path):
            with open(incident_summary_path, "r") as f:
                incident_data = json.load(f)
            st.markdown(f"""
            * **Event ID:** `{incident_data['event_id']}`
            * **Event Status / Class:** `{incident_data['event_status']}` (Confidence: `{incident_data['status_confidence']:.1%}`)
            * **Event Type:** `{incident_data['event_type']}` (Confidence: `{incident_data['type_confidence']:.1%}`)
            """)
            blackbox_df = pd.read_csv(blackbox_csv_path)
            st.dataframe(blackbox_df.head(10))
        else:
            st.info("Continuous rolling 15s pre-event buffer active.")

    with col_wit:
        st.markdown("#### 🛰️ Connected Vehicle Witness Discovery")
        # Check for accident report json files
        output_files = os.listdir("output") if os.path.exists("output") else []
        accident_reports = [f for f in output_files if f.startswith("accident_report_") and f.endswith(".json")]
        
        if accident_reports:
            latest_report = sorted(accident_reports)[-1]
            with open(os.path.join("output", latest_report), "r") as f:
                rep = json.load(f)
            st.success(f"📍 Geofence Witness Discovery Report (`{latest_report}`)")
            st.json(rep.get("witness_discovery", {}))
        else:
            st.info("Witness finder standby: Upon collision impact trigger, nearby connected vehicles will be discovered via simulated GPS Geofence.")

else:
    st.info("Waiting for telemetry data...")
