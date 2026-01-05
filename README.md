# ⚽ 3D Football Trajectory Tracker & Real-Time Mapper

This project is a computer vision pipeline. It transforms a standard 2D RGB video stream into a 3D spatial analysis tool,
 tracking a football's position, distance, and velocity.

## 🌟 Live Demo
[LINK TO YOUR DEPLOYED STREAMLIT APP]

## 🚀 Key Features
- **AI Detection:** Uses **YOLOv8** for robust ball tracking under varying light and motion blur.
- **3D Reconstruction:** Implements the **Pinhole Camera Model** to derive Z-depth using physical object constraints.
- **Egomotion Compensation:** Uses **ORB feature matching** and **Homography** to stabilize the 2D top-view map, making the trajectory independent of camera shake.
- **State Estimation:** Integrates a **Kalman Filter** to smooth trajectories and predict positions during high-speed motion or occlusions.

## 📐 The Technical "Story"

### 1. From Pixels to Meters (3D Math)
By knowing the horizontal field of view (HFOV) of the Intel RealSense D435i ($69.4^\circ$) and the physical diameter of a Size 5 football ($22\text{cm}$), we calculate the depth ($Z$) using:
$$f_x = \frac{\text{width}}{2 \times \tan(\text{HFOV}/2)}$$
$$Z = \frac{f_x \times \text{Diameter}_{real}}{\text{width}_{pixel}}$$



### 2. Camera Stabilization
To solve the "most challenging part" of the challenge, I implemented a feature-tracking loop. By calculating the movement of the background between frames, we apply a transformation matrix to the ball's coordinates. This ensures that even if the camera pans or shakes, the resulting **2D Top-View Map** shows the ball's true path relative to the ground.



## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/yourusername/football-tracker.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the local dashboard: `streamlit run app.py`

## 📊 Outputs
- **Annotated Video:** Real-time distance and velocity overlays.
- **Excel Report:** Comprehensive log of $X, Y, Z$ and $V$.
- **Stabilized Plot:** A top-view 2D trajectory map.
