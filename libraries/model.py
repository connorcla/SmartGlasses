import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

#OLED Screen
import time
from PIL import Image, ImageDraw, ImageFont
from luma.oled.device import ssd1306
from luma.core.interface.serial import spi
from luma.core.render import canvas
import subprocess

def print_to_screen(text):
    print("screen")
    serial = spi(port=0, device=0, gpio_DC=25, gpio_RST=27, gpio_CS=8)
    disp = ssd1306(serial, rotate=1)
    disp.clear()
    
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if the word can fit in the current line
        if len(current_line + " " + word) <= 9:
            current_line = current_line + " " + word if current_line else word
        else:
            # If the word can't fit, start a new line
            lines.append(current_line)
            current_line = word

    # Add the last line
    if current_line:
        lines.append(current_line)
    
    with canvas(disp) as draw:
        font = ImageFont.load_default()
        x_position = 10
        for i, line in enumerate(lines):
            y_position = 10 + (i*10)
            draw.text((x_position, y_position), line, font=font, fill=255)

    time.sleep(3)
    disp.clear()
    
    
def capture_image(filename="captured_image.jpg"):
	try:
		subprocess.run(["rpicam-still", "--immediate", "-o", filename], check=True)
		print("Image captured successfully")
	except subprocess.CalledProcessError:
		print("Error: Failed to capture image.")
		exit()
        

# Load color data from CSV
def load_color_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Name', 'Red', 'Green', 'Blue']]
    df['Red'] = df['Red'].str.rstrip('%').astype('float') / 100
    df['Green'] = df['Green'].str.rstrip('%').astype('float') / 100
    df['Blue'] = df['Blue'].str.rstrip('%').astype('float') / 100
    return df

# Train the color prediction model using KNN
def train_color_model(df):
    X = df[['Red', 'Green', 'Blue']]  # Features: Red, Green, Blue
    y = df['Name']  # Target: color name
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y_encoded)
    
    return knn, label_encoder

# Function to calculate the average RGB value from a square region of interest
def get_average_rgb(frame, center, size=25):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y = center
    half_size = size // 2
    roi = frame_rgb[y-half_size:y+half_size, x-half_size:x+half_size]
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color_normalized = avg_color / 255.0  # Normalize to range [0, 1]
    return avg_color_normalized

# Function to predict the color based on RGB values using the KNN model
def predict_color(knn_model, label_encoder, rgb_values, color_df, custom_thresholds):
    rgb_values = np.array(rgb_values).reshape(1, -1)  # Reshape for prediction
    predicted_label = knn_model.predict(rgb_values)  # Predict the color label
    predicted_color = label_encoder.inverse_transform(predicted_label)  # Get the color name
    
    # Check if the predicted color is within custom threshold range
    for i, row in color_df.iterrows():
        color_name = row['Name']
        ref_rgb = np.array([row['Red'], row['Green'], row['Blue']])
        diff = np.linalg.norm(rgb_values - ref_rgb)
        
        # Compare against custom thresholds
        if diff <= custom_thresholds.get(color_name, 0.1):
            return color_name  # Return the closest color based on the threshold
    
    return predicted_color[0]  # Fallback to predicted color if no match within threshold

# Main Program
def main():
    # Load color data from CSV
    color_df = load_color_data('sample-colors.csv')
    
    # Define custom color thresholds (for color "cusps")
    custom_thresholds = {
        'Red': 0.05,
        'Yellow': 0.05,
        'Blue': 0.05,
        # You can add more custom thresholds for other colors if needed
    }
    
    # Train the KNN model
    knn_model, label_encoder = train_color_model(color_df)
    
    # Open camera feed
    #cap = cv2.VideoCapture(0)
    #if not cap.isOpened():
    #    print("Error: Could not open video stream.")
    #    exit()
    
    print("Starting Calibration Mode...")
    print("Press '1' for Red, '2' for Yellow, '3' for Blue")

    red_value = None
    yellow_value = None
    blue_value = None
    
    # Predefined reference colors from the CSV
    red_ref = color_df[color_df['Name'] == 'Red'][['Red', 'Green', 'Blue']].values[0]
    yellow_ref = color_df[color_df['Name'] == 'Yellow'][['Red', 'Green', 'Blue']].values[0]
    blue_ref = color_df[color_df['Name'] == 'Blue'][['Red', 'Green', 'Blue']].values[0]
    
    for i in range(3):
        #ret, frame = cap.read()
        #if not ret:
        #    print("Failed to grab frame.")
        #    break
        
        capture_image("captured_image.jpg")
        frame = cv2.imread("captured_image.jpg")
        
        # Get the center of the frame
        height, width, _ = frame.shape
        center = (width // 2, height // 2)

        # Draw a crosshair on the frame
        half_size = 25 // 2
        cv2.rectangle(frame, (center[0] - half_size, center[1] - half_size),
                      (center[0] + half_size, center[1] + half_size), (0, 255, 0), 2)

        # Display live feed and instructions
        cv2.putText(frame, "Press 1 for Red, 2 for Yellow, 3 for Blue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Calibration', frame)

        # Wait for key press to capture the color
        key = cv2.waitKey(1) & 0xFF

        # Capture Red, Yellow, and Blue based on key press
        if i == 0:
            red_value = get_average_rgb(frame, center, size=25)
            print(f"Red Value Captured: {red_value}")
        elif i == 1:
            yellow_value = get_average_rgb(frame, center, size=25)
            print(f"Yellow Value Captured: {yellow_value}")
        elif i == 2:
            blue_value = get_average_rgb(frame, center, size=25)
            print(f"Blue Value Captured: {blue_value}")

        # Exit when all values have been captured
        if red_value is not None and yellow_value is not None and blue_value is not None:
            print("Calibration Complete!")
            break
    
    # Compare and adjust the captured values based on CSV references
    print("Adjusting Color Values based on Calibration...")
    print("Red value: ", red_value, " Yellow value: ", yellow_value, " Blue value: ", blue_value)
    
    # Calculate differences between captured and reference values
    red_diff = red_value - red_ref
    yellow_diff = yellow_value - yellow_ref
    blue_diff = blue_value - blue_ref

    while True:
        #ret, frame = cap.read()
        #if not ret:
        #    print("Failed to grab frame.")
        #    break
        
        capture_image("captured_image.jpg")
        frame = cv2.imread("captured_image.jpg")

        # Get the center of the frame
        height, width, _ = frame.shape
        center = (width // 2, height // 2)

        # Get the average RGB value of the 25x25 region around the crosshair
        avg_rgb = get_average_rgb(frame, center, size=25)

        # Adjust the predicted color values based on calibration
        adjusted_rgb = [
            avg_rgb[0] + red_diff[0],
            avg_rgb[1] + yellow_diff[1],
            avg_rgb[2] + blue_diff[2]
        ]

        # Predict the color based on average RGB
        predicted_color = predict_color(knn_model, label_encoder, adjusted_rgb, color_df, custom_thresholds)
        
        # Display the predicted color on the frame
        cv2.putText(frame, f'Color: {predicted_color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        text_to_screen = predicted_color
        print_to_screen(text_to_screen)
        
        # Define the position for the color square on the frame
        color_square_x = 10
        color_square_y = 70

        # Convert the adjusted RGB values into a NumPy array, scale them to 8-bit RGB, and ensure the format
        color_square = np.array(adjusted_rgb) * 255
        color_square = np.clip(color_square, 0, 255)  # Ensure values are within [0, 255]
        color_square = color_square.astype(int)  # Convert to integer values

        # Convert RGB to BGR (OpenCV uses BGR format)
        color_square_bgr = (color_square[2], color_square[1], color_square[0])  # Reversed order for BGR

        # Debug: Check the values before drawing the rectangle
        # print("Color Square BGR:", color_square_bgr)

        # Ensure color_square_bgr is a valid tuple with 3 integers
        color_square_bgr = tuple(map(int, color_square_bgr))  # Force integer values

        # Print the values being passed to cv2.rectangle
        #print("Drawing Rectangle with Top-Left Corner:", (color_square_x, color_square_y))
        #print("Drawing Rectangle with Bottom-Right Corner:", (color_square_x + 50, color_square_y + 50))
        #print("Color for the rectangle:", color_square_bgr)

         # Draw a crosshair on the frame
        half_size = 25 // 2
        cv2.rectangle(frame, (center[0] - half_size, center[1] - half_size),
                      (center[0] + half_size, center[1] + half_size), (0, 255, 0), 2)

        # Validate 'frame' again
        if isinstance(frame, np.ndarray) and frame.ndim == 3:
            #print("Frame is valid. Shape:", frame.shape)
            
            # Draw the rectangle
            cv2.rectangle(frame, (color_square_x, color_square_y),
                        (color_square_x + 50, color_square_y + 50), color_square_bgr, -1)
        #else:
            #print("Error: 'frame' is not a valid NumPy array or does not have 3 dimensions.")

        # Show the image
        cv2.imshow('Live Color Prediction', frame)

        # Exit condition (press 'Enter' to quit)
        if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter key
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
