import cv2
import numpy as np
import subprocess

def categorize_color(r, g, b):
    if r < 50 and g < 50 and b < 50:
        return "black / gray"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    elif 80 < r < 150 and 50 < g < 100 and 50 < b < 100:
        return "brown"
    elif r > 120 and g < 80 and b < 80:
        return "red"
    elif r > 150 and g > 100 and b < 80:
        return "orange"
    elif r > 150 and g > 150 and b < 150:
        return "yellow"
    elif g > 180 and r < 150 and b < 150:
        return "green"
    elif r > 100 and b < 45 and g < 50:
        return "violet"
    elif r > 50 and b > 150 and g < 150:
        return "blue"
    else:
        return "unknown"
        
def get_color_rgb(r,g,b):
    hsv = cv2.cvtColor(np.uint8([[[r,g,b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h,s,v = hsv
    
    if s < 50:
        return "gray"
    elif v < 50:
        return "black"
    elif 0 <= h < 10 or 170 <= h <= 180:
        return "red"
    elif 10 <= h < 40:
        return "orange"
    elif 40 <= h < 90:
        return "yellow"
    elif 90 <= h < 170:
        return "green"
    elif 170 <= h < 260:
        return "blue"
    elif 260 <= h < 320:
        return "purple"
    else:
        return "unknown"
        

def capture_image(filename="captured_image.jpg"):
    try:
        subprocess.run(["rpicam-still", "-o", filename], check=True)
        print("Image captured successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to capture image with rpicam-still.")
        exit()

#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#if not cap.isOpened():
#    print("Error: Could not open camera.")
#    exit()

while True:
    capture_image("captured_image.jpg")
    frame = cv2.imread("captured_image.jpg")
    
    #if not ret:
    #    print("Failed to capture image")
    #    break

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    
    # Define a small 5x5 region around the center
    crosshair_size = 2  # Half the width of the crosshair square
    roi = frame[center_y - crosshair_size:center_y + crosshair_size + 1, center_x - crosshair_size:center_x + crosshair_size + 1]
    
    # Compute the mean RGB values
    mean_bgr = np.mean(roi, axis=(0, 1))
    mean_r, mean_g, mean_b = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
    
    # Categorize the color based on mean RGB values
    color_category = categorize_color(mean_r, mean_g, mean_b)
    print(f"Mean RGB: ({mean_r}, {mean_g}, {mean_b}) - Color Category: {color_category}")
    
    # Draw a small crosshair at the center
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Show the frame
    cv2.imshow("Camera Feed", frame)
    
    # Exit if Enter key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        print("Exit condition triggered: Enter key pressed.")
        break

cap.release()
cv2.destroyAllWindows()
