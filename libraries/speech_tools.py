import sys
import speech_recognition as sr
import sounddevice
import threading
import io
import datetime

from libraries.oled_print_tools import *

def check_sentence_dur(audio, min_duration=0.25):
    audio_stream = io.BytesIO(audio.get_wav_data())
    import wave
    with wave.open(audio_stream, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return duration >= min_duration
        

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
                
            if not check_sentence_dur(audio):
                return timer
    
            try:
                text = rec.recognize_google(audio, language="en-EN")
                #print(f"{text}", end = " ")
                text = text[0].upper() + text[1:]
                sys.stdout.write(text)
                sys.stdout.write(". ")
                sys.stdout.flush()
                
                if text == "Time":
                    now = datetime.datetime.now()
                    now_str = now.strftime("%H:%M %m-%d-%Y")
                    return print_to_screen("Current time:\n\n" + now_str, timer)

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
        exit()
        return timer
    
    return timer
            

def start_listening(timer):
    return recognize_speech(timer)
