#Speech recognition
import sys
import speech_recognition as sr
import sounddevice
import threading

#OLED Screen
import time
from PIL import Image, ImageDraw, ImageFont
from luma.oled.device import ssd1306
from luma.core.interface.serial import spi
from luma.core.render import canvas

rec = sr.Recognizer()
rec.energy_threshold = 400

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

def recognize_speech():
    while True:
        try:
            with sr.Microphone() as source:
                rec.adjust_for_ambient_noise(source, duration=1)
                #print("Listening...")
                try:
                    audio = rec.listen(source, timeout=1, phrase_time_limit=10)
                except sr.WaitTimeoutError:
                    #print("Timeout")
                    continue
        
                try:
                    text = rec.recognize_google(audio, language="en-EN")
                    #print(f"{text}", end = " ")
                    text = text[0].upper() + text[1:]
                    sys.stdout.write(text)
                    sys.stdout.write(". ")
                    sys.stdout.flush()
                    print_to_screen(text)
                except sr.UnknownValueError:
                    #print("Google Web Speech API could not understand the audio")
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results from Google Web Speech API; {e}")
        except KeyboardInterrupt:
            print("Exiting program")
            return
            

def start_listening():
    recognize_speech()
            

if __name__ == "__main__":
    mic_list = sr.Microphone.list_microphone_names()
    #for i, name in enumerate(mic_list):
    #    print(f"Microphone: {name} and device {i}")
    #recognize_speech()
    start_listening()
