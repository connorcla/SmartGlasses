import RPi.GPIO as GPIO
import time
import subprocess

from libraries.state_machine_tools import *
from libraries.speech_tools import *
from libraries.oled_print_tools import *
from libraries.asl_tools import *


class Glasses_State_Machine(StateMachine):
    def InitExtension(self):
        self.AddState(InitState("init_state", self))
        self.AddState(SystemOff("system_off", self))
        self.AddState(Caption("caption", self))
        self.AddState(Color("color", self))
        self.AddState(ASL("asl", self))
        self.AddState(OnTransition("on_transition", self))
        self.AddState(OffTransition("off_transition", self))
        self.AddState(CapColTransition("cap_col", self))
        self.AddState(ColASLTransition("col_asl", self))
        self.AddState(ASLCapTransition("asl_cap", self))

        self.curr_state: State = self.GetState("init_state")

    
    def Execute(self):
        if(self.is_working):
            match self.curr_state.name:
                case "init_state":
                    self.curr_state = self.GetState("system_off")
                case "system_off":
                    if GPIO.input(power_btn):
                        self.curr_state = self.GetState("on_transition")
                    else:
                        self.curr_state = self.GetState("system_off")
                case "caption":
                    if GPIO.input(power_btn) and not GPIO.input(mode_btn):
                        self.curr_state = self.GetState("off_transition")
                        self.GetState("caption").timer = 0
                    elif not GPIO.input(power_btn) and GPIO.input(mode_btn):
                        self.curr_state = self.GetState("cap_col")
                        self.GetState("caption").timer = 0
                    else:
                        self.curr_state = self.GetState("caption")
                case "color":
                    if GPIO.input(power_btn) and not GPIO.input(mode_btn):
                        self.curr_state = self.GetState("off_transition")
                    elif not GPIO.input(power_btn) and GPIO.input(mode_btn):
                        self.curr_state = self.GetState("col_asl")
                    else:
                        self.curr_state = self.GetState("color")
                case "asl":
                    if GPIO.input(power_btn) and not GPIO.input(mode_btn):
                        self.curr_state = self.GetState("off_transition")
                        self.GetState("asl").asl_timer = 0
                    elif not GPIO.input(power_btn) and GPIO.input(mode_btn):
                        self.curr_state = self.GetState("asl_cap")
                        self.GetState("asl").asl_timer = 0
                    else:
                        self.curr_state = self.GetState("asl")
                case "on_transition":
                    if GPIO.input(power_btn):
                        self.curr_state = self.GetState("on_transition")
                    else:
                        self.curr_state = self.GetState("caption")
                case "off_transition":
                    if GPIO.input(power_btn):
                        self.curr_state = self.GetState("off_transition")
                    else:
                        self.curr_state = self.GetState("system_off")
                case "cap_col":
                    if GPIO.input(mode_btn):
                        self.curr_state = self.GetState("cap_col")
                    else:
                        self.curr_state = self.GetState("color")
                case "col_asl":
                    if GPIO.input(mode_btn):
                        self.curr_state = self.GetState("col_asl")
                    else:
                        self.curr_state = self.GetState("asl")
                case "asl_cap":
                    if GPIO.input(mode_btn):
                        self.curr_state = self.GetState("asl_cap")
                    else:
                        self.curr_state = self.GetState("caption")
                case _:
                    self.curr_state = self.GetState("init_state")

            self.RunState(self.curr_state.name)


class InitState(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.PrintInit)
    
    def PrintInit(self):
        print("Initializing")


class SystemOff(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.TurnOff)
    
    def TurnOff(self):
        GPIO.output(power_led, GPIO.LOW)
        GPIO.output(red_led, GPIO.LOW)
        GPIO.output(green_led, GPIO.LOW)
        GPIO.output(blue_led, GPIO.LOW)


class Caption(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.SpeechLoop)
        
        self.timer = 0

    def SpeechLoop(self):
        GPIO.output(power_led, GPIO.HIGH)
        GPIO.output(red_led, GPIO.LOW)
        GPIO.output(green_led, GPIO.LOW)
        GPIO.output(blue_led, GPIO.HIGH)

        # Speech Recognition Loop ToDo PUT IN
        self.timer = start_listening(self.timer)


class Color(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.ColorLoop)

    def ColorLoop(self):
        GPIO.output(power_led, GPIO.HIGH)
        GPIO.output(red_led, GPIO.HIGH)
        GPIO.output(green_led, GPIO.LOW)
        GPIO.output(blue_led, GPIO.LOW)

        # Camera Loop ToDo PUT IN


class ASL(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.ASLLoop)
        self.asl_timer = 0

    def ASLLoop(self):
        GPIO.output(power_led, GPIO.HIGH)
        GPIO.output(red_led, GPIO.LOW)
        GPIO.output(green_led, GPIO.HIGH)
        GPIO.output(blue_led, GPIO.LOW)
        
        letter = "default"

        # ASL Loop ToDo PUT IN
        if(self.asl_timer == 0):
            CaptureImage()
            label = asl_nn_model.PredictShort(model_type, input_path, transform_type)
            letter = ConvertLabelToChar(label)
        
        self.asl_timer = print_to_screen(letter, self.asl_timer)



class OnTransition(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.PrintOnTransition)

    def PrintOnTransition(self):
        clear_screen(0)
        print("On transition")


class OffTransition(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.PrintOffTransition)

    def PrintOffTransition(self):
        clear_screen(0)
        print("Off transition")


class CapColTransition(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.PrintCapColTransition)

    def PrintCapColTransition(self):
        clear_screen(0)
        print("CapCol transition")


class ColASLTransition(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.PrintColASLTransition)

    def PrintColASLTransition(self):
        clear_screen(0)
        print("ColASL transition")


class ASLCapTransition(State):
    def InitExtension(self):
        self.action_manager.AddAction(self.PrintASLCapTransition)

    def PrintASLCapTransition(self):
        clear_screen(0)
        print("ASLCap transition")



if __name__ == "__main__":

    # --------------------------------------------------

    torch.manual_seed(1)

    path_manager: MainPathManager = MainPathManager("main_model_path_manager")

    tag_type: str                      = "Default"
    transform_type: str                = "Default"
    model_type: str                    = "Default"
    model_num: str                     = "35.0.11"
    model_path: str = path_manager.GetLiteralDataPath("model_path") + model_num + ".pth"
    input_path: str = path_manager.GetLiteralDataPath("image_path")                       

    asl_nn_model: NNASL = NNASL("asl_nn_model")
    asl_nn_model.LoadModel(model_type, model_path, tag_type)  

    # --------------------------------------------------

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    power_led = 5
    red_led = 6
    green_led = 13
    blue_led = 19
    
    power_btn = 20
    mode_btn = 21
    
    GPIO.setup(power_led, GPIO.OUT)
    GPIO.setup(red_led, GPIO.OUT)
    GPIO.setup(green_led, GPIO.OUT)
    GPIO.setup(blue_led, GPIO.OUT)
    GPIO.setup(power_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(mode_btn, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
  
    
    glasses_sm = Glasses_State_Machine("glasses_state_machine", 10)
    glasses_sm.Start()
    while True:
        glasses_sm.Execute()
        time.sleep(0.01)
