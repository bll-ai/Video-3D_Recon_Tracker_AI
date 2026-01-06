import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import time

st.set_page_config(page_title="3D Football Tracker", layout="wide")
st.title("⚽ 3D Football Trajectory")

# Auto-locate the 12MB video
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "rgb.avi")

col1, col2 = st.columns([2, 1])
with col1: video_feed = st.empty()
with col2: 
    st.write("### 3D Top-View (Meters)")
    chart_container = st.empty()

if os.path.exists(VIDEO_PATH) and st.sidebar.button("🚀 Start Demo"):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # D435i Specs for Math
    W = int(cap.get(3))
    fx = W / (2 * np.tan(np.deg2rad(69.4 / 2)))
    
    # history stores X and Z. Pre-fill with limits to lock the map scale (stops jumps).
    history = pd.DataFrame({'Lateral (X)': [-6, 6], 'Depth (Z)': [0, 35]})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=0.35, classes=[32], verbose=False)
        if results[0].boxes:
            b = results[0].boxes[0].xyxy[0].cpu().numpy()
            z = (fx * 0.22) / (b[2] - b[0]) # Depth
            x = (((b[0] + b[2]) / 2) - (W / 2)) * z / fx # Lateral
            
            if 1 < z < 35:
                new_pt = pd.DataFrame({'Lateral (X)': [x], 'Depth (Z)': [z]})
                history = pd.concat([history, new_pt], ignore_index=True)
            
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

        # FAST RENDERING
        video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        # Native scatter_chart is hardware-accelerated and won't flicker
        if not history.empty:
            chart_container.scatter_chart(history, x='Lateral (X)', y='Depth (Z)', height=450)
            
        time.sleep(0.01)
    cap.release()
