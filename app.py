import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import time

st.set_page_config(page_title="3D Tracker", layout="wide")
st.title("⚽ 3D Football Trajectory")

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "rgb.avi")

col1, col2 = st.columns([2, 1])
with col1: video_feed = st.empty()
with col2: map_chart = st.empty()

if os.path.exists(VIDEO_PATH) and st.sidebar.button("🚀 Start Demo"):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(VIDEO_PATH)
    fx = int(cap.get(3)) / (2 * np.tan(np.deg2rad(69.4 / 2)))
    
    # Persistent data for the map
    history = pd.DataFrame(columns=['x', 'z'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=0.35, classes=[32], verbose=False)
        if results[0].boxes:
            b = results[0].boxes[0].xyxy[0].cpu().numpy()
            z = (fx * 0.22) / (b[2] - b[0])
            x = (((b[0] + b[2]) / 2) - (int(cap.get(3)) / 2)) * z / fx
            
            # Filter jumps and add to history
            if 1 < z < 30:
                history = pd.concat([history, pd.DataFrame({'x':[x], 'z':[z]})], ignore_index=True)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

        # FAST RENDERING
        video_feed.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        if not history.empty:
            # Native scatter chart is much smoother than Matplotlib
            map_chart.scatter_chart(history, x='x', y='z', height=400)
            
        time.sleep(0.01)
    cap.release()
