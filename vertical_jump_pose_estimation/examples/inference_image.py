from ultralytics import YOLO
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = YOLO("yolo11s-pose.pt")

# Load image
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQO9oV4P4jBdgbVXwJ2UwdvL8zUQh7FNAkckg&s"
image = imutils.url_to_image(url)

# Run inference
results = model(image)

# Define COCO keypoint skeleton (pairs of connected joints)
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
]

# Draw keypoints + skeleton
for result in results:
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()  # shape: (num_people, num_keypoints, 2)
        for person in keypoints:
            # Draw skeleton connections
            for (i, j) in skeleton:
                if i < len(person) and j < len(person):
                    x1, y1 = person[i]
                    x2, y2 = person[j]
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# Show in matplotlib (so colors are correct)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
