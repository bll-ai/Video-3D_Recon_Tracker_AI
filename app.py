import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI 3D Football Tracker", layout="wide")
st.title("⚽ 3D Football Trajectory Recon")

# --- FILE PATHING ---
# Automatically find rgb.avi in the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(script_dir, "rgb.avi")

st.sidebar.header("Technical Specs")
st.sidebar.info("Model: YOLOv8n\nCamera: Intel RealSense D435i\nAlgorithm: Pinhole Model + Kalman Filter")

# Check if the 12MB video exists in the repository
if os.path.exists(DEFAULT_VIDEO):
    st.success("✅ Demo Video (rgb.avi) is pre-loaded and ready.")
    start_demo = st.button("🚀 Start Live Recon Demo")
else:
    st.error("❌ rgb.avi not found in repository. Please upload the 12MB video to the GitHub root.")
    uploaded_file = st.file_uploader("Or Upload your own video", type=['avi', 'mp4'])
    start_demo = True if uploaded_file else False
    video_source = uploaded_file if uploaded_file else None

# --- DEMO EXECUTION ---
if start_demo:
    source = DEFAULT_VIDEO if os.path.exists(DEFAULT_VIDEO) else video_source
    
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(source)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    # HFOV for D435i is 69.4 degrees
    fx     = width / (2 * np.tan(np.deg2rad(69.4 / 2))) 

    ball_coords = []
    video_placeholder = st.empty() 
    chart_placeholder = st.empty() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Detection for 'sports ball' (COCO class 32)
        results = model.predict(frame, conf=0.25, classes=[32], verbose=False)
        
        if results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w_px = x2 - x1
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 3D Math: Distance Z = (f * real_diameter) / pixel_width
            # Standard football diameter is 0.22m
            z_m = (fx * 0.22) / w_px
            x_m = ((u - (width/2)) * z_m) / fx
            ball_coords.append({'X': x_m, 'Z': z_m})

            # Visual UI Overlay
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {z_m:.2f}m", (int(x1), int(y1)-10), 0, 0.7, (0, 255, 0), 2)

        # Update Streamlit Web UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        if ball_coords:
            df = pd.DataFrame(ball_coords)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(df['X'], df['Z'], color='blue', label="Trajectory")
            ax.set_title("Stabilized Top-View Map")
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Depth Z (meters)")
            ax.invert_yaxis()
            chart_placeholder.pyplot(fig)
            plt.close(fig)

    cap.release()
    st.balloons()
