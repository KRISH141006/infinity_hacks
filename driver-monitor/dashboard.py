import streamlit as st
import json
import os
import sys
import pandas as pd
import altair as alt
import subprocess
import cv2
import time
from datetime import datetime

# Absolute directory of this dashboard script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path_str):
    """Ensures relative paths are resolved against BASE_DIR regardless of launch cwd."""
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(BASE_DIR, path_str))

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
    .badge-card {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85em;
        margin: 4px 2px;
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
st.markdown('<div class="cockpit-sub">Multi-Signal Temporal Driver Monitoring System (DMS) & Advanced Road Perception (ADAS)</div>', unsafe_allow_html=True)

# Configuration options in sidebar
st.sidebar.header("🔧 DMS Config (Internal Camera)")

default_driver_in = "input/tired_driver.webm" if os.path.exists(resolve_path("input/tired_driver.webm")) else "input/driver.mp4"
default_driver_out = "output/tired_driver_processed.mp4" if os.path.exists(resolve_path("output/tired_driver_processed.mp4")) else "output/processed_driver.mp4"

input_video_path = st.sidebar.text_input("Internal Input Video", default_driver_in)
processed_video_path = st.sidebar.text_input("Internal Processed Video", default_driver_out)
telemetry_path = st.sidebar.text_input("Driver Telemetry JSON", "output/telemetry.json")

if st.sidebar.button("👤 Process Driver DMS Video"):
    with st.spinner("Processing driver monitoring video with multi-signal DMS engine..."):
        script_path = os.path.join(BASE_DIR, "process_video.py")
        result = subprocess.run([
            sys.executable, script_path, 
            "--input", resolve_path(input_video_path),
            "--output", resolve_path(processed_video_path),
            "--telemetry", resolve_path(telemetry_path)
        ], cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Driver DMS video processed successfully!")
            st.rerun()
        else:
            st.sidebar.error(f"Error processing driver video: {result.stderr or result.stdout}")

st.sidebar.markdown("---")
st.sidebar.header("🛣️ ADAS Config (External Camera)")
input_road_path = st.sidebar.text_input("External Road Video", "input/external_road.mp4")
processed_road_path = st.sidebar.text_input("External Processed Video", "output/processed_road.mp4")
road_telemetry_path = st.sidebar.text_input("Road Telemetry JSON", "output/road_telemetry.json")

if st.sidebar.button("🛣️ Process Road ADAS Video"):
    with st.spinner("Processing external camera labeled video & tracking..."):
        script_path = os.path.join(BASE_DIR, "process_road_video.py")
        result = subprocess.run([
            sys.executable, script_path,
            "--input", resolve_path(input_road_path),
            "--output", resolve_path(processed_road_path),
            "--telemetry", resolve_path(road_telemetry_path)
        ], cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("External road video labeled & processed successfully!")
            st.rerun()
        else:
            st.sidebar.error(f"Error processing road video: {result.stderr or result.stdout}")

# Mode Selector Tab
tab_recorded, tab_live = st.tabs(["📊 Recorded Analytics & Sync Explorer", "🔴 Live Webcam DMS Mode"])

with tab_live:
    st.markdown("### 🔴 Live Real-Time Driver Monitoring Stream")
    st.info("Continuous live webcam inference using MediaPipe face mesh, adaptive normalized EAR, head pose velocity, and attention fusion.")
    live_start = st.toggle("Start Webcam DMS Stream", value=False)
    
    if live_start:
        from src.live_stream import LiveDMSStream
        live_engine = LiveDMSStream(model_dir=resolve_path("models"))
        cam = cv2.VideoCapture(0)
        frame_window = st.image([])
        live_stats_placeholder = st.empty()
        
        stop_btn = st.button("Stop Live Stream")
        
        while live_start and not stop_btn:
            ret, live_frame = cam.read()
            if not ret:
                st.error("Cannot access webcam.")
                break
            
            live_rec, lms = live_engine.process_frame(live_frame)
            
            # Draw overlay on live frame
            h, w = live_frame.shape[:2]
            overlay = live_frame.copy()
            cv2.rectangle(overlay, (15, 15), (380, 240), (10, 15, 25), -1)
            cv2.addWeighted(overlay, 0.70, live_frame, 0.30, 0, live_frame)
            cv2.putText(live_frame, f"STATE: {live_rec['master_driver_state']}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if "ALERT" in live_rec['master_driver_state'] else (0, 0, 255), 2)
            cv2.putText(live_frame, f"Attention Index: {live_rec['attention_score']}/100", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(live_frame, f"Drowsiness: {live_rec['drowsiness_score']} | Fatigue: {live_rec['fatigue_score']}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.putText(live_frame, f"Distraction: {live_rec['distraction_score']} [{live_rec['distraction_state']}]", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.putText(live_frame, f"EAR: {live_rec['ear']:.2f} (Norm:{live_rec['normalized_ear']:.2f}) | Gaze: {live_rec['gaze_state']}", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.putText(live_frame, f"Head: {live_rec['head_pose_state']} (P:{live_rec['pitch']:.0f}° V:{live_rec['head_pitch_velocity']:.0f}°/s)", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.putText(live_frame, f"Phone: {'DETECTED 📱' if live_rec['phone_usage'] else 'NONE'}", (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255) if live_rec['phone_usage'] else (200, 200, 200), 1)

            # Display frame
            frame_rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.01)

        cam.release()

with tab_recorded:
    # Check files availability using absolute resolved paths
    abs_driver_video = resolve_path(processed_video_path)
    abs_driver_telemetry = resolve_path(telemetry_path)
    abs_road_video = resolve_path(processed_road_path)
    abs_road_telemetry = resolve_path(road_telemetry_path)

    has_driver_video = os.path.exists(abs_driver_video)
    has_driver_telemetry = os.path.exists(abs_driver_telemetry)
    has_road_video = os.path.exists(abs_road_video)
    has_road_telemetry = os.path.exists(abs_road_telemetry)

    # Load Driver Telemetry
    driver_telemetry = None
    if has_driver_telemetry:
        try:
            with open(abs_driver_telemetry, "r") as f:
                driver_telemetry = json.load(f)
        except Exception as e:
            st.error(f"Error loading driver telemetry: {e}")

    # Load Road Telemetry
    road_telemetry = None
    if has_road_telemetry:
        try:
            with open(abs_road_telemetry, "r") as f:
                road_telemetry = json.load(f)
        except Exception as e:
            st.error(f"Error loading road telemetry: {e}")

    if not has_driver_video and not has_road_video:
        st.info("💡 Please process the driver video or road video using the sidebar buttons.")

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

        # -------------------------------------------------------------
        # 1. GLOBAL SAFETY STATUS BANNER
        # -------------------------------------------------------------
        master_state = curr_driver.get("master_driver_state", "ALERT")
        collision_risk = curr_road.get("collision_risk", "Low")
        road_hazard = curr_road.get("road_hazard", 0) or curr_driver.get("hazard", 0)

        if "DROWSY" in master_state or collision_risk == "High":
            st.error(f"🔴 CRITICAL SAFETY ALERT: Master State: {master_state} | Collision Risk: {collision_risk.upper()}")
        elif master_state in ["DISTRACTED", "PHONE_USAGE", "YAWNING", "FATIGUED"] or collision_risk == "Medium" or road_hazard:
            st.warning(f"🟡 ADAS WARNING: Driver State: {master_state} | Hazard: {'YES' if road_hazard else 'NO'} | Risk: {collision_risk.upper()}")
        else:
            st.success("🟢 DRIVER ALERT & ACTIVE: Normal Secure Drive")

        # -------------------------------------------------------------
        # 2. FOUR CORE METRIC GAUGES (DMS KPI)
        # -------------------------------------------------------------
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            att_val = curr_driver.get("attention_score", 100)
            st.metric("Driver Attention Index", f"{att_val} / 100", delta=None)
        with kpi2:
            drowsy_val = curr_driver.get("drowsiness_score", 0)
            st.metric("Drowsiness Score", f"{drowsy_val} / 100", delta=curr_driver.get("drowsiness_state", "ALERT"), delta_color="inverse")
        with kpi3:
            fatigue_val = curr_driver.get("fatigue_score", 0)
            st.metric("Fatigue Score", f"{fatigue_val} / 100", delta=curr_driver.get("fatigue_state", "ALERT"), delta_color="inverse")
        with kpi4:
            distract_val = curr_driver.get("distraction_score", 0)
            st.metric("Distraction Score", f"{distract_val} / 100", delta=curr_driver.get("distraction_state", "ATTENTIVE"), delta_color="inverse")

        # -------------------------------------------------------------
        # 3. DRIVER STATUS BADGES
        # -------------------------------------------------------------
        st.markdown(f"""
        <div style="background-color: #111827; padding: 12px 18px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 15px;">
            <span class="badge-card" style="background-color: #1e3a8a; color: #bfdbfe;">HEAD: {curr_driver.get('head_pose_state', 'FORWARD')} (Y:{curr_driver.get('yaw', 0.0):.0f}° P:{curr_driver.get('pitch', 0.0):.0f}°)</span>
            <span class="badge-card" style="background-color: #1e3a8a; color: #bfdbfe;">GAZE: {curr_driver.get('gaze_state', 'CENTER')} (Conf:{curr_driver.get('gaze_confidence', 1.0):.2f})</span>
            <span class="badge-card" style="background-color: {'#7f1d1d' if curr_driver.get('eye_closed_duration', 0) > 0.25 else '#064e3b'}; color: {'#fca5a5' if curr_driver.get('eye_closed_duration', 0) > 0.25 else '#a7f3d0'};">EYES: {curr_driver.get('eye_state', 'OPEN')} ({curr_driver.get('eye_closed_duration', 0.0):.2f}s)</span>
            <span class="badge-card" style="background-color: {'#7c2d12' if curr_driver.get('is_yawning', False) else '#1e3a8a'}; color: {'#fed7aa' if curr_driver.get('is_yawning', False) else '#bfdbfe'};">MOUTH: {'YAWNING' if curr_driver.get('is_yawning', False) else 'NORMAL'} (MAR:{curr_driver.get('mar', 0.15):.2f})</span>
            <span class="badge-card" style="background-color: {'#7f1d1d' if curr_driver.get('phone_usage', False) else '#064e3b'}; color: {'#fca5a5' if curr_driver.get('phone_usage', False) else '#a7f3d0'};">PHONE: {'DETECTED 📱' if curr_driver.get('phone_usage', False) else 'NOT DETECTED'}</span>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 4. DUAL CAMERA STREAM PLAYBACK
        # -------------------------------------------------------------
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.markdown("#### 📹 Internal Camera (Driver Monitoring System)")
            if has_driver_video:
                st.video(abs_driver_video)
            else:
                st.warning("Processed driver video not found. Run processing in the sidebar.")
                
        with col_cam2:
            st.markdown("#### 🛣️ External Camera (Road Perception - Driver's POV)")
            if has_road_video:
                st.video(abs_road_video)
            else:
                st.info("Processed labeled road video will appear here once processed.")

        # -------------------------------------------------------------
        # 5. EXPANDABLE DEBUG PANEL (REQUIREMENT 24)
        # -------------------------------------------------------------
        with st.expander("🔬 Granular Signal Debug & Evidence Breakdown Panel", expanded=False):
            st.markdown("##### 🧪 Real-Time Extracted Feature Telemetry")
            dbg_c1, dbg_c2, dbg_c3 = st.columns(3)
            
            with dbg_c1:
                st.markdown("**👁️ Eye & Blink Metrics**")
                st.write(f"• EAR: `{curr_driver.get('ear', 0.28):.3f}` (L: `{curr_driver.get('ear_left', 0.28):.3f}` | R: `{curr_driver.get('ear_right', 0.28):.3f}`)")
                st.write(f"• EAR Baseline: `{curr_driver.get('ear_baseline', 0.28):.3f}` (Thresh: `{curr_driver.get('ear_threshold', 0.20):.3f}`)")
                st.write(f"• Normalized EAR: `{curr_driver.get('normalized_ear', 1.0):.2f}`")
                st.write(f"• Eye Closure Duration: `{curr_driver.get('eye_closed_duration', 0.0):.2f} s`")
                st.write(f"• Blink Count / Rate: `{curr_driver.get('blink_count', 0)}` blinks (`{curr_driver.get('blink_rate_per_min', 0)}`/min)")

            with dbg_c2:
                st.markdown("**🧠 Fatigue & Yawn Dynamics**")
                st.write(f"• PERCLOS (Short 2-5s): `{curr_driver.get('perclos_short', 0.0):.2f}`")
                st.write(f"• PERCLOS (Med 10-20s): `{curr_driver.get('perclos_medium', 0.0):.2f}`")
                st.write(f"• PERCLOS (Long 30-60s): `{curr_driver.get('perclos_long', 0.0):.2f}`")
                st.write(f"• Mouth MAR: `{curr_driver.get('mar', 0.15):.3f}` (Vel: `{curr_driver.get('mar_velocity', 0.0):.2f}`)")
                st.write(f"• Yawn State / Prob: `{curr_driver.get('yawn_state', 'NO_YAWN')}` (`{curr_driver.get('yawn_probability', 0.0):.2f}`)")

            with dbg_c3:
                st.markdown("**📐 Head Kinematics & Gaze Deviation**")
                st.write(f"• Raw Euler Angles: `Y:{curr_driver.get('raw_yaw', 0.0):.1f}° P:{curr_driver.get('raw_pitch', 0.0):.1f}° R:{curr_driver.get('raw_roll', 0.0):.1f}°`")
                st.write(f"• Neutral Calib: `Y:{curr_driver.get('neutral_yaw', 0.0):.1f}° P:{curr_driver.get('neutral_pitch', 0.0):.1f}° R:{curr_driver.get('neutral_roll', 0.0):.1f}°`")
                st.write(f"• Pitch Velocity: `{curr_driver.get('head_pitch_velocity', 0.0):.1f}°/s` (Nodding: `{curr_driver.get('is_nodding', False)}`)")
                st.write(f"• Gaze Offsets: `dx:{curr_driver.get('gaze_dx', 0.0):.3f} dy:{curr_driver.get('gaze_dy', 0.0):.3f}`")
                st.write(f"• Face Tracking Quality: `{curr_driver.get('tracking_quality', 1.0):.2f}` (`{curr_driver.get('tracking_state', 'TRACKED')}`)")

            st.markdown("##### ⚖️ Drowsiness Evidence Breakdown")
            st.json(curr_driver.get("evidence_breakdown", {}))

        # -------------------------------------------------------------
        # 6. ALERTS & TELEMETRY WAVEFORMS
        # -------------------------------------------------------------
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

        # Include Road Events if present
        road_events_path = resolve_path("output/road_events.json")
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
            st.subheader("📈 Multi-Signal Temporal Waveforms")
            if df_driver is not None and not df_driver.empty:
                chart_df = df_driver[["timestamp_sec", "attention_score", "drowsiness_score", "fatigue_score", "distraction_score"]].copy()
                chart_df.columns = ["Time (s)", "Attention Score", "Drowsiness Score", "Fatigue Score", "Distraction Score"]
                melted_df = chart_df.melt("Time (s)", var_name="Metric", value_name="Score")
                
                line_chart = alt.Chart(melted_df).mark_line().encode(
                    x='Time (s):Q',
                    y='Score:Q',
                    color='Metric:N'
                ).properties(height=230)
                st.altair_chart(line_chart, use_container_width=True)

        # -------------------------------------------------------------
        # 7. BLACK BOX & WITNESS DISCOVERY
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("📦 RoadGuardian Black Box & Geofence Witness Discovery")
        col_bb, col_wit = st.columns(2)
        
        with col_bb:
            st.markdown("#### 🔒 Incident Data Recorder (EDR)")
            blackbox_csv_path = resolve_path("output/roadguardian_blackbox_event.csv")
            incident_summary_path = resolve_path("output/roadguardian_incident_summary.json")
            
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
            output_dir = resolve_path("output")
            output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
            accident_reports = [f for f in output_files if f.startswith("accident_report_") and f.endswith(".json")]
            
            if accident_reports:
                latest_report = sorted(accident_reports)[-1]
                with open(os.path.join(output_dir, latest_report), "r") as f:
                    rep = json.load(f)
                st.success(f"📍 Geofence Witness Discovery Report (`{latest_report}`)")
                st.json(rep.get("witness_discovery", {}))
            else:
                st.info("Witness finder standby: Upon collision impact trigger, nearby connected vehicles will be discovered via simulated GPS Geofence.")
    else:
        st.info("Waiting for telemetry data...")
