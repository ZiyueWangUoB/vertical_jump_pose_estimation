from ultralytics import YOLO
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vertical_jump_pose_estimation.examples.helpers import draw_points_on_img

# Load model
model = YOLO("yolo11s-pose.pt")

video_path = "isaiah_jump.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()  # ret = boolean (frame available), frame = image
    if not ret:
        print("End of video or cannot fetch the frame.")
        break

    results = model(frame)
    frame = draw_points_on_img(results, frame)

    # Show the frame
    cv2.imshow("Video", frame)

    # Press 'q' to exit early
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()