import torch
import cv2
import numpy as np
import mediapipe as mp

from typing import List
import os

import torch
import torchvision
from torchvision import transforms, datasets, models

import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import time
import random

import mediapipe as mp
import cv2

# Example image tensor (simulating a (C, H, W) tensor)

transform = transforms.ToTensor()
img = transform(Image.open("./input/G_test.jpg"))
# img = torch.randn(3, 200, 200)  # Random tensor, shape (C, H, W)

# Scale to [0, 255] range and ensure the tensor is in byte format (uint8)
img = (img * 255).clamp(0, 255).byte()  # Now the tensor is in uint8 format

# Convert tensor to numpy array with shape (H, W, C)
img = img.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
img = cv2.resize(img, (224,224))

# Show the image after converting from tensor (before color conversion)
cv2.imshow("Image after Tensor to NumPy", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert from BGR (OpenCV default) to RGB (MediaPipe expects RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show the image after BGR to RGB conversion
cv2.imshow("Image after BGR to RGB", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
    # Process the image with MediaPipe
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        print(f"Detected {len(results.multi_hand_landmarks)} hands.")
    else:
        print("No hands detected.")

    # Draw landmarks if any hands are detected
    marked_img = img_rgb.copy()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                marked_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=1, circle_radius=1)
            )

    # Show the image after detecting landmarks
    cv2.imshow("Image with Hand Landmarks", marked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
