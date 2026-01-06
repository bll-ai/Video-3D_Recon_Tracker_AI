import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="AI 3D Football Tracker", layout="wide")
st.title("⚽ 3D Football Trajectory Recon")

script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(script_dir, "rgb.avi")

# --- UI CONTAINERS ---
col1, col2 = st.columns([2, 1])
with col1:
    video_placeholder = st.empty()
with col2:
    st.write("### 2D Top-View Map")
    chart_placeholder = st.empty()

# --- INITIALIZATION ---
if os.path.exists(DEFAULT_VIDEO):
    if st.button("🚀 Start Live Recon Demo"):
        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(DEFAULT_VIDEO)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fx = width / (2 * np.tan(np.deg2rad(69.4 / 2)))

        ball_coords = []
        
        # --- PROCESSING LOOP ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Detection
            results = model.predict(frame, conf=0.25, classes=[32], verbose=False)
            
            if results[0].boxes:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w_px = x2 - x1
                u, v = (x1 + x2) / 2, (y1 + y2) / 2
                
                # 3D Depth Math
                z_m = (fx * 0.22) / w_px
                x_m = ((u - (width/2)) * z_m) / fx
                
                # Store for mapping
                ball_coords.append({'X': x_m, 'Z': z_m})

                # Visuals
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Z: {z_m:.2f}m", (int(x1), int(y1)-10), 0, 0.7, (0, 255, 0), 2)

            # 2. Update Video (The fix for the "Fixed Image")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # 3. Update Map (The fix for the "Wrong Map")
            if len(ball_coords) > 1:
                df = pd.DataFrame(ball_coords)
                fig, ax = plt.subplots(figsize=(4, 5))
                # We use X for horizontal and Z for depth (distance from camera)
                ax.plot(df['X'], df['Z'], color='blue', linewidth=2, marker='o', markersize=2)
                ax.set_xlim(-5, 5)  # Constrain X so the map doesn't jump
                ax.set_ylim(0, 20)  # Constrain Z (depth)
                ax.set_xlabel("Lateral (Meters)")
                ax.set_ylabel("Depth (Meters)")
                ax.invert_yaxis()   # Flip so distance increases 'up' the screen
                ax.grid(True, linestyle='--', alpha=0.6)
                chart_placeholder.pyplot(fig)
                plt.close(fig)

            # Small delay to let Streamlit's frontend catch up
            # Without this, it looks like a static image because it processes too fast
            time.sleep(0.01)

        cap.release()
        st.balloons()
