import mediapipe as mp
import cv2
import os

import sys 
sys.path.append("./input")


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

if not os.path.exists("./input/G_test.jpg"):
      raise FileNotFoundError(f"Path is not valid.")

with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
  img = cv2.imread("./input/G_test.jpg")
  if img is None:
    print("Image not loaded")
  img = cv2.flip(img, 1)
  results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  marked_img = img.copy()

  for hand_landmarks in results.multi_hand_landmarks:          
    mp_drawing.draw_landmarks(marked_img, hand_landmarks,
                              mp_hands.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 255, 0), thickness=2, circle_radius=1),
                              connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                color=(0, 0, 255), thickness=1, circle_radius=1)
                              )
          
  cv2.imshow("marked image", marked_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

