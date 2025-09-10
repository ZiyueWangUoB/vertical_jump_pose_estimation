import cv2
import numpy as np
from ultralytics.engine.results import Results

skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),        # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12),              # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
    ]

def draw_points_on_img(results: Results, image: np.array):
    """
        Draw keypoints and skeleton on the image from inference results.
    """

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
    return image


def detect_center_of_gravity(keypoints: np.array):
    """
        Center of gravity will be defined as the mean of the keypoints of shoulder hips and ankles
    """
    relevant_indices = [5, 6, 11, 12, 13, 14]  # shoulders, hips, ankles
    relevant_keypoints = keypoints[relevant_indices]
    center_of_gravity = np.mean(relevant_keypoints, axis=0)
    return center_of_gravity

def draw_center_of_gravity(results: Results, image: np.array):
    """
        Draw center of gravity on the image.
    """
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # shape: (num_people, num_keypoints, 2)
            for person in keypoints:
                cog = detect_center_of_gravity(person)
                cv2.circle(image, (int(cog[0]), int(cog[1])), 5, (0, 255, 0), -1)  # Green dot for COG
    return image