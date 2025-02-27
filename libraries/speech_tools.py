import sys
import speech_recognition as sr
import sounddevice
import threading

from libraries.oled_print_tools import *

def recognize_speech(timer):

    rec = sr.Recognizer()
    rec.energy_threshold = 400
    
    print(timer)

    if timer != 0:
        return print_to_screen("", timer)

    try:
        with sr.Microphone() as source:
            rec.adjust_for_ambient_noise(source, duration=1)
            #print("Listening...")
            try:
                audio = rec.listen(source, timeout=0.5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                #print("Timeout")
                return timer
    
            try:
                text = rec.recognize_google(audio, language="en-EN")
                #print(f"{text}", end = " ")
                text = text[0].upper() + text[1:]
                sys.stdout.write(text)
                sys.stdout.write(". ")
                sys.stdout.flush()

                #Use OLED library
                return print_to_screen(text, timer)
            except sr.UnknownValueError:
                #print("Google Web Speech API could not understand the audio")
                pass
            except sr.RequestError as e:
                print(f"Could not request results from Google Web Speech API; {e}")
            except KeyboardInterrupt:
                print("Exiting program")
                return timer
    except KeyboardInterrupt:
        print("Exiting program")
        return timer
    
    return timer
            

def start_listening(timer):
    return recognize_speech(timer)
