import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import subprocess

from libraries.oled_print_tools import *
    
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
    
def print_color(timer):
	
	if timer != 0:
		return print_to_screen("", timer)
	
	# Load color data from CSV
	color_df = load_color_data('./libraries/sample-colors.csv')
	
	# Define custom color thresholds (for color "cusps")
	custom_thresholds = {'Red': 0.05, 'Yellow': 0.05, 'Blue': 0.05}
	
	# Train the KNN model
	knn_model, label_encoder = train_color_model(color_df)
	
	#FOR NOW CHANGE LATER no calibration yet
	red_ref = color_df[color_df['Name'] == 'Red'][['Red', 'Green', 'Blue']].values[0]
	yellow_ref = color_df[color_df['Name'] == 'Yellow'][['Red', 'Green', 'Blue']].values[0]
	blue_ref = color_df[color_df['Name'] == 'Blue'][['Red', 'Green', 'Blue']].values[0]
	
	red_value = [6.37595316e-01, 2.35457516e-01, 1.36165577e-05]
	yellow_value = [0.70857843, 0.84159858, 0.14859749]
	blue_value = [0.,         0.23517157, 0.62379493]
	
	red_diff = red_value - red_ref
	yellow_diff = yellow_value - yellow_ref
	blue_diff = blue_value - blue_ref
	
	capture_image("captured_image.jpg")
	frame = cv2.imread("captured_image.jpg")
	
	height, width, _ = frame.shape
	center = (width // 2, height // 2)
	
	avg_rgb = get_average_rgb(frame, center, size=25)
	
	adjusted_rgb = [avg_rgb[0] + red_diff[0], avg_rgb[1] + yellow_diff[1], avg_rgb[2] + blue_diff[2]]
	
	predicted_color = predict_color(knn_model, label_encoder, adjusted_rgb, color_df, custom_thresholds)
	
	cv2.putText(frame, f'Color: {predicted_color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	text_to_screen = predicted_color
	return print_to_screen(text_to_screen, timer)
        
        
#For calibration and full references, see model.py
