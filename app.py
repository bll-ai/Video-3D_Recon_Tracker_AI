import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import time

# --- 1. SETTINGS & SPECS ---
st.set_page_config(page_title="3D Recon Tracker", layout="wide")
st.title("⚽ AI 3D Football Reconstruction")

# Intel RealSense D435i Specs
HFOV = 69.4
BALL_DIA = 0.22 

script_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(script_dir, "rgb.avi")

# --- 2. LAYOUT ---
col_vid, col_map = st.columns([2, 1])

with col_vid:
    st.subheader("AI Vision Feed")
    video_feed = st.empty()

with col_map:
    st.subheader("Stabilized 2D Map (Meters)")
    # Using a native Streamlit chart for speed and stability
    chart_container = st.empty()

# --- 3. THE ENGINE ---
if os.path.exists(VIDEO_PATH):
    if st.sidebar.button("🚀 Launch Live Demo"):
        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fx = width / (2 * np.tan(np.deg2rad(HFOV / 2)))
        
        # We store the trajectory in a simple DataFrame for the chart
        trajectory_df = pd.DataFrame(columns=['Lateral (X)', 'Depth (Z)'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = model.predict(frame, conf=0.3, classes=[32], verbose=False)
            
            if results[0].boxes:
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                w_px = box[2] - box[0]
                u = (box[0] + box[2]) / 2
                
                # Math: Distance Calculation
                z_m = (fx * BALL_DIA) / w_px
                x_m = ((u - (width / 2)) * z_m) / fx
                
                # Sanity check: Only map if distance is realistic (0-40m)
                if 0.5 < z_m < 40:
                    new_point = pd.DataFrame({'Lateral (X)': [x_m], 'Depth (Z)': [z_m]})
                    trajectory_df = pd.concat([trajectory_df, new_point], ignore_index=True)

                # Draw Overlay on Frame
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # --- RENDER FEED ---
            video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # --- RENDER NATIVE CHART ---
            # This is 10x faster than Matplotlib and won't jitter
            if not trajectory_df.empty:
                chart_container.scatter_chart(
                    trajectory_df,
                    x='Lateral (X)',
                    y='Depth (Z)',
                    x_label="Meters (Left/Right)",
                    y_label="Meters (Depth)",
                    use_container_width=True
                )

            # Small sleep to yield to the browser's thread
            time.sleep(0.01)

        cap.release()
        st.success("Reconstruction Finished!")
else:
    st.error("rgb.avi not found in repository.")
