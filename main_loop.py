import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

power_led = 29
mode_led = 31

power_btn = 26
mode_btn = 24

GPIO.setup(power_led, GPIO.OUT)
GPIO.setup(mode_led, GPIO.OUT)
GPIO.setup(power_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(mode_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

state = 0

def switch_states():
	global state
	match state:
		case 0:
			state = 1
		case 1:
			if GPIO.input(26):
				state = 2
			else:
				state = 1
		case 2:
			if GPIO.input(26):
				state = 2
			else:
				state = 4
		case 3:
			if GPIO.input(26):
				state = 3
			else:
				state = 1
		case 4:
			if not GPIO.input(26) and not GPIO.input(24):
				state = 4
			elif GPIO.input(26) and not GPIO.input(24):
				state = 3
			else:
				state = 5
		case 5:
			if GPIO.input(24):
				state = 5
			else:
				state = 7
		case 6: 
			if GPIO.input(24):
				state = 6
			else:
				state = 4
		case 7:
			if not GPIO.input(26) and not GPIO.input(24):
				state = 7
			elif GPIO.input(26) and not GPIO.input(24):
				state = 3
			else:
				state = 6
		case _:
			state = 0
			
def activate_state():
	global state
	match state:
		case 0:
			return
		case 1:
			GPIO.output(29, GPIO.LOW)
			GPIO.output(31, GPIO.LOW)
		case 2:
			return
		case 3:
			return
		case 4:
			GPIO.output(29, GPIO.HIGH)
			GPIO.output(31, GPIO.LOW)
		case 5:
			return
		case 6:
			return
		case 7:
			GPIO.output(29, GPIO.HIGH)
			GPIO.output(31, GPIO.HIGH)
		case _:
			return
	

if __name__ == "__main__":
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(False)

	power_led = 29
	mode_led = 31

	power_btn = 26
	mode_btn = 24

	GPIO.setup(29, GPIO.OUT)
	GPIO.setup(31, GPIO.OUT)
	GPIO.setup(26, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
	GPIO.setup(24, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
	
	state = 0
	
	while True:
		switch_states()
		activate_state()
		print(state)
		time.sleep(0.1)
		#if GPIO.input(26) == GPIO.HIGH:
			#print("Button")

