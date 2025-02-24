import RPi.GPIO as GPIO
import time


state = 0

def switch_states():
	global state
	match state:
		case 0:
			state = 1
		case 1:
			if GPIO.input(power_btn):
				state = 2
			else:
				state = 1
		case 2:
			if GPIO.input(power_btn):
				state = 2
			else:
				state = 4
		case 3:
			if GPIO.input(power_btn):
				state = 3
			else:
				state = 1
		case 4:
			if not GPIO.input(power_btn) and not GPIO.input(mode_btn):
				state = 4
			elif GPIO.input(power_btn) and not GPIO.input(mode_btn):
				state = 3
			else:
				state = 5
		case 5:
			if GPIO.input(mode_btn):
				state = 5
			else:
				state = 7
		case 6: 
			if GPIO.input(mode_btn):
				state = 6
			else:
				state = 4
		case 7:
			if not GPIO.input(power_btn) and not GPIO.input(mode_btn):
				state = 7
			elif GPIO.input(power_btn) and not GPIO.input(mode_btn):
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
			GPIO.output(power_led, GPIO.LOW)
			GPIO.output(mode_led, GPIO.LOW)
		case 2:
			return
		case 3:
			return
		case 4:
			GPIO.output(power_led, GPIO.HIGH)
			GPIO.output(mode_led, GPIO.LOW)
		case 5:
			return
		case 6:
			return
		case 7:
			GPIO.output(power_led, GPIO.HIGH)
			GPIO.output(mode_led, GPIO.HIGH)
		case _:
			return
	

if __name__ == "__main__":
	
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(False)
	
	power_led = 29
	mode_led = 31
	#pin 33 also open for GPIO led
	
	power_btn = 38
	mode_btn = 40

	
	GPIO.setup(power_led, GPIO.OUT)
	GPIO.setup(mode_led, GPIO.OUT)
	GPIO.setup(power_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
	GPIO.setup(mode_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
	state = 0
	
	while True:
		switch_states()
		activate_state()
		print(state)
		time.sleep(0.1)
		#if GPIO.input(26) == GPIO.HIGH:
			#print("Button")

