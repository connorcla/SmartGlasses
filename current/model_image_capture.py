import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ImageFont
from luma.oled.device import ssd1306
from luma.core.interface.serial import spi
from luma.core.render import canvas

def print_to_screen(text):
    serial = spi(port=0, device=0, gpio_DC=25, gpio_RST=27, gpio_CS=8)
    disp = ssd1306(serial, rotate=1)
    disp.clear()
    
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= 9:
            current_line = current_line + " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    
    with canvas(disp) as draw:
        font = ImageFont.load_default()
        x_position = 10
        for i, line in enumerate(lines):
            y_position = 10 + (i * 10)
            draw.text((x_position, y_position), line, font=font, fill=255)

def load_color_data(file_path):
    df = pd.read_csv(file_path)
    df[['Red', 'Green', 'Blue']] = df[['Red', 'Green', 'Blue']].astype(float) / 255
    return df

def train_color_model(df):
    X = df[['Red', 'Green', 'Blue']]
    y = df['Name']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y_encoded)
    return knn, label_encoder

def get_average_rgb(image, center, size=25):
    x, y = center
    half_size = size // 2
    roi = image[y-half_size:y+half_size, x-half_size:x+half_size]
    avg_color = np.mean(roi, axis=(0, 1)) / 255.0
    return avg_color

def predict_color(knn_model, label_encoder, rgb_values):
    rgb_values = np.array(rgb_values).reshape(1, -1)
    predicted_label = knn_model.predict(rgb_values)
    return label_encoder.inverse_transform(predicted_label)[0]

def main():
    color_df = load_color_data('sample-colors.csv')
    knn_model, label_encoder = train_color_model(color_df)
    
    image = cv2.imread('image.png')
    if image is None:
        print("Error: Could not load image.png")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    
    avg_rgb = get_average_rgb(image_rgb, center)
    predicted_color = predict_color(knn_model, label_encoder, avg_rgb)
    
    print(f"Predicted Color: {predicted_color}")
    print_to_screen(predicted_color)

if __name__ == "__main__":
    main()
