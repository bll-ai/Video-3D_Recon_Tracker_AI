import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="3D Football Tracker Demo", layout="wide")
st.title("⚽ 3D Football Trajectory & Velocity Tracker")
st.markdown("Upload a video to see AI-powered 3D tracking and camera-stabilized mapping in action.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("AI Confidence", 0.1, 1.0, 0.25)
ball_dia = st.sidebar.number_input("Ball Diameter (meters)", value=0.22)

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload your video (avi/mp4)", type=['avi', 'mp4'])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Setup Logic
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fx = width / (2 * np.tan(np.deg2rad(69.4 / 2)))

    # Storage
    ball_data = []
    
    # UI Progress
    progress_bar = st.progress(0)
    frame_display = st.empty() # Container for the video stream
    
    # Logic Loop
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Run AI (simplified for speed in web demo)
        results = model.predict(frame, conf=conf_threshold, classes=[32], verbose=False)
        
        if results[0].boxes:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w_px = x2 - x1
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 3D Depth Math
            z_m = (fx * ball_dia) / w_px
            x_m = ((u - (width/2)) * z_m) / fx
            
            # Draw on frame for the UI
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(frame, f"Dist: {z_m:.2f}m", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ball_data.append({'Time': frame_count/fps, 'X': x_m, 'Z': z_m})

        # Update Live Preview in Web App
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, channels="RGB", use_container_width=True)
        progress_bar.progress(frame_count / total_frames)

    # --- FINAL RESULTS ---
    st.success("Processing Complete!")
    if ball_data:
        df = pd.DataFrame(ball_data)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trajectory Data")
            st.dataframe(df)
            
        with col2:
            st.subheader("Top-View Plot")
            fig, ax = plt.subplots()
            ax.plot(df['X'], df['Z'], '-o')
            ax.set_title("Ball Movement (Z is Depth)")
            ax.invert_yaxis()
            st.pyplot(fig)