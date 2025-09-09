from ultralytics import YOLO
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vertical_jump_pose_estimation.examples.helpers import draw_points_on_img

# Load model
model = YOLO("yolo11s-pose.pt")

# Load image
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQO9oV4P4jBdgbVXwJ2UwdvL8zUQh7FNAkckg&s"
image = imutils.url_to_image(url)

# Run inference
results = model(image)
image = draw_points_on_img(results, image)

# Show in matplotlib (so colors are correct)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
