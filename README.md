# Cricket Ball Tracking & Trajectory Prediction with YOLO11

A computer vision system designed to detect, track, and predict the trajectory of a cricket ball in real-time. This project utilizes the YOLO11 architecture for object detection and BoT-SORT for persistent multi-object tracking, with custom physics-based trajectory forecasting.

## 🚀 Features

- Custom YOLO11 Model: Fine-tuned to detect small, high-speed cricket balls that standard COCO models often miss.
- Persistent Tracking: Uses BoT-SORT to maintain ball ID even during brief occlusions or high-velocity motion.
- Parabolic Trajectory Prediction: Uses polynomial regression to forecast the ball's path, accounting for gravity.
- Bounce Detection: Logic to identify sudden changes in vertical velocity to flag ball bounces.
- Visual Analytics: Real-time overlays for "comet-tail" history, future path prediction, and angle calculation.

## 🛠️ Requirements

- Python 3.9+
- Ultralytics YOLO11
- OpenCV (cv2)
- NumPy

```bash
pip install ultralytics opencv-python numpy
```
