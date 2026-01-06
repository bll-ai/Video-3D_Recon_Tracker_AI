import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import time

# --- 1. PAGE & UI INITIALIZATION ---
st.set_page_config(page_title="3D Football Recon Demo", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00ff00; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚽ Live AI 3D Football Reconstruction")

# Path to your 12MB video in the GitHub repo
script_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(script_dir, "rgb.avi")

# Layout: Video on the left, Stabilized Map on the right
col_vid, col_map = st.columns([3, 2])

with col_vid:
    st.subheader("AI Vision Feed")
    video_frame = st.empty()

with col_map:
    st.subheader("Stabilized 3D Ground Map")
    map_plot = st.empty()
    metrics = st.empty()

# --- 2. THE PROCESSING ENGINE ---
if os.path.exists(VIDEO_PATH):
    # Auto-start logic: No need for user to upload anything
    if st.sidebar.button("🚀 Start Live Demo", use_container_width=True):
        
        # Load YOLOv8 Nano (optimized for web speed)
        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Camera Specs: Intel RealSense D435i
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        HFOV   = 69.4
        fx     = width / (2 * np.tan(np.deg2rad(HFOV / 2)))
        
        # Data Buffers
        trajectory_x = []
        trajectory_z = []
        
        # Kalman Filter initialization for smoothing (stops the map from jittering)
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # AI Inference
            results = model.predict(frame, conf=0.25, classes=[32], verbose=False)
            
            z_m, x_m = 0.0, 0.0
            
            if results[0].boxes:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w_px = x2 - x1
                u, v = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Pinhole Math: Depth (Z) and Lateral (X)
                z_m = (fx * 0.22) / w_px  # 0.22m = standard football
                x_m = ((u - (width / 2)) * z_m) / fx
                
                # Kalman Smoothing
                kf.correct(np.array([[np.float32(x_m)], [np.float32(z_m)]]))
                predict = kf.predict()
                x_m, z_m = predict[0][0], predict[1][0]

                trajectory_x.append(x_m)
                trajectory_z.append(z_m)

                # Draw Overlay
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(frame, f"Depth: {z_m:.1f}m", (int(x1), int(y1)-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- UPDATE UI ---
            # 1. Update Video Feed (RGB Conversion for Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame.image(frame_rgb, channels="RGB")

            # 2. Update 2D Map (Fixed Scale for Professional Look)
            if len(trajectory_x) > 1:
                fig, ax = plt.subplots(figsize=(5, 7))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1e2129')
                
                # Plot trajectory
                ax.plot(trajectory_x, trajectory_z, color='#00ff00', linewidth=2, marker='o', markersize=3, alpha=0.7)
                
                # Fixed Axis Limits: Prevents the "jumping/wrong map" feeling
                ax.set_xlim(-6, 6)    # 6 meters left/right
                ax.set_ylim(0, 30)    # 30 meters depth
                
                # Visual Styling
                ax.invert_yaxis()
                ax.set_xlabel("Lateral Offset (m)", color='white')
                ax.set_ylabel("Distance from Camera (m)", color='white')
                ax.tick_params(colors='white')
                ax.grid(True, linestyle='--', alpha=0.3)
                
                map_plot.pyplot(fig)
                plt.close(fig)

                # 3. Update Metrics Dashboard
                metrics.markdown(f"""
                | Metric | Value |
                | :--- | :--- |
                | **Current Depth** | {z_m:.2f} m |
                | **Lateral Pos** | {x_m:.2f} m |
                | **Frames Tracked** | {len(trajectory_x)} |
                """)

            # Control playback speed for the browser
            time.sleep(0.01)

        cap.release()
        st.success("🏁 Reconstruction Completed Successfully.")
        st.balloons()
else:
    st.error(f"Missing File: Could not find 'rgb.avi' at {VIDEO_PATH}. Please ensure it is uploaded to your GitHub repository.")
