from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



def angle_between_lines(m1, m2=1):
    if m1 != -1 / m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0


class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)

    def add(self, item):
        self.queue.append(item)

    def pop(self):
        self.queue.popleft()

    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue

    def __len__(self):
        return len(self.queue)


model = YOLO("runs/detect/train9/weights/best.pt")

video_name = "test1"
video_path = "../home_balldrop_vids/" + video_name + ".mp4"
output_path = '../home_balldrop_vids/white_ball/60fps/out/output_kf_' + video_name + '.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
else:
    print("Video opened successfully. Starting tracking...")

ret = True
prevTime = 0
centroid_history = FixedSizeQueue(10)
start_time = time.time()
interval = 0.6
paused = False
angle = 0
prev_frame_time = 0
new_frame_time = 0

# --- BEFORE THE WHILE LOOP ---
# Get original video properties for the writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

# 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 10 # Adjust for smoothness

# Storage for plotting
time_steps = []
measured_x, measured_y = [], []
kf_predicted_x, kf_predicted_y = [], []
naive_predicted_x, naive_predicted_y = [], []
frame_count = 0

while ret:
    ret, frame = cap.read()


    if ret:



        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        #print(list(centroid_history.queue))
        current_time = time.time()
        if current_time - start_time >= interval and len(centroid_history) > 0:
            centroid_history.pop()
            start_time = current_time

        results = model.track(frame,
                              persist=True,
                              conf=0.6,
                              verbose=False,
                              tracker="botsort.yaml",
                              imgsz=1280)
        boxes = results[0].boxes
        box = boxes.xyxy
        rows, cols = box.shape
        centroid_x, centroid_y = 0,0
        naive_pred_x, naive_pred_y = 0,0
        # 1. Prediction Step: Predict where the ball is moving
        predicted = kf.predict()
        pred_x, pred_y = int(predicted[0]), int(predicted[1])

        if len(box) != 0:
            for i in range(rows):


                x1, y1, x2, y2 = box[i]
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()

                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

                centroid_history.add((centroid_x, centroid_y))
                cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # 2. Correction Step: Update the filter with the actual YOLO detection
                measured = np.array([[centroid_x], [centroid_y]], np.float32)
                kf.correct(measured)

        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            for i in range(1, len(centroid_history)):
                # if math.sqrt(y_diff**2+x_diff**2)<7:
                cv2.line(frame, centroid_history.get_queue()[i - 1], centroid_history.get_queue()[i], (255, 0, 0), 4)

        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())

            x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            # Calculate average velocity over the last few points
            avg_x_diff = (centroid_list[-1][0] - centroid_list[-2][0]) / len(centroid_list)
            avg_y_diff = (centroid_list[-1][1] - centroid_list[-2][1]) / len(centroid_list)

            if (x_diff != 0):
                m1 = y_diff / x_diff
                if m1 == 1:
                    angle = 90
                elif m1 != 0:
                    angle = 90 - angle_between_lines(m1)
                if angle >= 45:
                    print("ball bounced")

            # Naive Predictions
            future_positions = [centroid_list[-1]]
            print("Naive Future Positions: ", future_positions[0])  # Naive
            naive_pred_x, naive_pred_y = future_positions[0]

            for i in range(1, 5):
                future_positions.append(
                    (
                        centroid_list[-1][0] + x_diff * i,
                        centroid_list[-1][1] + y_diff * i
                    )
                )
            for i in range(1, len(future_positions)):
                cv2.line(frame, future_positions[i - 1], future_positions[i], (0, 255, 0), 4)
                cv2.circle(frame, future_positions[i], radius=3, color=(0, 0, 255), thickness=-1)

            # Kalman Predictions
            future_kf = kf.statePost.copy()
            kf_pred_x, kf_pred_y = int(future_kf[0]), int(future_kf[1])
            print(f"KF Future Positions: ({kf_pred_x}, {kf_pred_y})")  # KF
            for i in range(1, 5):
                # Manually apply the transition matrix for multi-step prediction
                future_kf = np.dot(kf.transitionMatrix, future_kf)
                f_x, f_y = int(future_kf[0]), int(future_kf[1])
                cv2.circle(frame, (f_x, f_y), 2, color=(255, 255, 255), thickness=-1)




        text = "Angle: {:.2f} degrees".format(angle)
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        frame_resized = cv2.resize(frame, (1000, 600))
        cv2.imshow('frame', frame_resized)
        #print(fps)
        # --- INSIDE THE LOOP, AFTER ALL DRAWING ---
        # Write the processed frame to the output file
        # Note: 'frame' here should be the original size version with drawings
        out.write(frame)

        # Display logic (resizing only for view, not for saving)
        frame_resized = cv2.resize(frame, (1000, 600))
        #cv2.imshow('frame', frame_resized)

        frame_count += 1
        time_steps.append(frame_count)

        # Save ground truth (YOLO measurement)
        measured_x.append(centroid_x)
        measured_y.append(centroid_y)

        # Save KF prediction
        kf_predicted_x.append(pred_x)
        kf_predicted_y.append(pred_y)

        # Save Naive prediction (from previous code)
        naive_predicted_x.append(naive_pred_x)
        naive_predicted_y.append(naive_pred_y)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):
            paused = not paused

            while paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    paused = not paused
                elif key == ord('q'):
                    break


# --- AFTER THE LOOP ---
cap.release()
out.release() # CRITICAL: must release the writer to save the file
cv2.destroyAllWindows()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot X-coordinate comparison
ax1.plot(time_steps, measured_x, 'ro', markersize=3, label='Actual (YOLO)')
ax1.plot(time_steps, kf_predicted_x, 'g-', label='Kalman Filter')
ax1.plot(time_steps, naive_predicted_x, 'b--', alpha=0.5, label='Naive Prediction')
ax1.set_ylabel('X Coordinate')
ax1.legend()
ax1.set_title('Tracking Performance: X and Y vs Time')

# Plot Y-coordinate comparison
ax2.plot(time_steps, measured_y, 'ro', markersize=3, label='Actual (YOLO)')
ax2.plot(time_steps, kf_predicted_y, 'g-', label='Kalman Filter')
ax2.plot(time_steps, naive_predicted_y, 'b--', alpha=0.5, label='Naive Prediction')
ax2.set_ylabel('Y Coordinate')
ax2.set_xlabel('Frame Number')
ax2.legend()

plt.tight_layout()
plt.show()