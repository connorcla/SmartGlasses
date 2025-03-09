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

  img_width, img_height, _ = img.shape
  x_min = img_width
  y_min = img_height
  x_max= 0
  y_max = 0

  marked_img = img.copy()
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:   
      # draw landmark       
      mp_drawing.draw_landmarks(marked_img, hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                  color=(0, 255, 0), thickness=1, circle_radius=1),
                                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                  color=(0, 0, 255), thickness=1, circle_radius=1)
                                )
      
      # update coords for cropping bounding box
      for landmark in hand_landmarks.landmark:
        curr_x = int(landmark.x * img_width)
        curr_y = int(landmark.y * img_height)
        x_min, y_min = min(curr_x, x_min), min(curr_y, y_min)
        x_max, y_max = max(curr_x, x_max), max(curr_y, y_max)

  
  # crop image with square ratio
  padding = 25
  x_length = x_max - x_min
  y_length = y_max - y_min
  longer_side = max(x_length, y_length)
  x_padding = int((longer_side - x_length) / 2) + padding
  y_padding = int((longer_side - y_length) / 2) + padding
  x_min = max(x_min - x_padding, 0)
  y_min = max(y_min - y_padding, 0)
  x_max = min(x_max + x_padding, img_width)
  y_max = min(y_max + y_padding, img_height)

  cropped_img = marked_img[x_min:x_max, y_min:y_max]
  cropped_img = cv2.resize(cropped_img, (128, 128))

  # display resulting images        
  cv2.imshow("marked image", marked_img)
  cv2.imshow("cropped image", cropped_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

