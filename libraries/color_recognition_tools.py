import cv2
import numpy as np
import pandas as pd
from collections import namedtuple
import math
import subprocess
import time

# Define a named tuple for color data
Color = namedtuple('Color', ['Name', 'Hex', 'R', 'G', 'B'])

# Load the CSV data into a pandas DataFrame
csv_data = pd.read_csv('sample-colors.csv')  # Change the path to your actual CSV file

# Convert percentage columns to integer values
def percent_to_int(value):
    return int(float(value.rstrip('%')))

# List of colors from the CSV file
colors = []
for _, row in csv_data.iterrows():
    name = row['Name']
    hex_value = row['Hex Triplet']
    r = percent_to_int(row['Red'])
    g = percent_to_int(row['Green'])
    b = percent_to_int(row['Blue'])
    colors.append(Color(name, hex_value, r, g, b))

# Function to calculate the Euclidean distance between two RGB colors
def color_distance(c1, c2):
    return math.sqrt((c1.R - c2.R) ** 2 + (c1.G - c2.G) ** 2 + (c1.B - c2.B) ** 2)

# Function to find the closest matching color from the CSV
def find_closest_color(rgb):
    closest_color = min(colors, key=lambda c: color_distance(c, Color("", "", rgb[0], rgb[1], rgb[2])))
    return closest_color
    
def capture_image(filename="captured_image.jpg"):
	try:
		subprocess.run(["rpicam-still", "--immediate", "-o", filename], check=True)
		print("Image captured successfully")
	except subprocess.CalledProcessError:
		print("Error: Failed to capture image.")
		exit()

# Initialize the webcam
#cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
#if not cap.isOpened():
#    print("Error: Could not open webcam.")
#    exit()

# Define the crosshair size
crosshair_size = 25
half_size = crosshair_size // 2

# Main loop
while True:
    capture_image("captured_image.jpg")
    frame = cv2.imread("captured_image.jpg")
    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Draw the crosshair in the center of the screen
    center_x, center_y = width // 2, height // 2
    cv2.rectangle(frame, (center_x - half_size, center_y - half_size), (center_x + half_size, center_y + half_size), (0, 255, 0), 2)

    # Crop the region of interest (crosshair area)
    crosshair_area = frame[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    # Calculate the average RGB value of the crosshair area
    avg_rgb = np.mean(crosshair_area, axis=(0, 1))  # Compute mean of the RGB channels

    # Convert the average RGB value to integer
    avg_rgb = tuple(map(int, avg_rgb))

    # Find the closest color match
    closest_color = find_closest_color(avg_rgb)

    # Display the closest color name and RGB on the frame
    color_info = f"{closest_color.Name} ({closest_color.Hex})"
    cv2.putText(frame, color_info, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame with the crosshair
    cv2.imshow("Webcam Feed with Crosshair", frame)
    
    print(color_info)
    time.sleep(2)

    # Check for key press to exit the loop (Enter key to exit)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
