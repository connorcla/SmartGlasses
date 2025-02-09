import sys
import speech_recognition as sr
import sounddevice

rec = sr.Recognizer()
rec.energy_threshold = 400


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
                except sr.UnknownValueError:
                    #print("Google Web Speech API could not understand the audio")
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results from Google Web Speech API; {e}")
        except KeyboardInterrupt:
            print("Exiting program")
            return
            

if __name__ == "__main__":
    mic_list = sr.Microphone.list_microphone_names()
    #for i, name in enumerate(mic_list):
    #    print(f"Microphone: {name} and device {i}")
    recognize_speech()
