import sys 
sys.path.append("../..")

from libraries.state_machine_tools import *



class ButtonStateMachine(StateMachine):
  def InitExtension(self): 
    self.AddState(InitializeState("initialize_state"))
    self.AddState(OffState("off_state"))
    self.AddState(OnState("on_state"))

    self.curr_state: State = self.GetState("initialize_state")

  def Execute(self):
    if (self.is_working):
      if (self.period_manager.SatisfiesPeriod()):
        match self.curr_state.name:
          case "initialize_state":
            self.curr_state = self.GetState("off_state")

          case "off_state":
            self.curr_state = self.GetState("on_state")

          case "on_state":
            self.curr_state = self.GetState("off_state")

        match self.curr_state.name:
          case "initialize_state":
            self.RunState("initialize_state")

          case "off_state":
            self.RunState("off_state")

          case "on_state":
            self.RunState("on_state")

    return


    

class InitializeState(State):
  def InitExtension(self):
    self.action_manager.AddAction(self.PrintInitialize)

  def PrintInitialize(self):
    print("Initializing state!")


class OffState(State):
  def InitExtension(self):
    self.action_manager.AddAction(self.PrintOff)

  def PrintOff(self):
    print("Off state!")


class OnState(State):
  def InitExtension(self):
    self.action_manager.AddAction(self.PrintOn)
  
  def PrintOn(self):
    print("On state!")




if __name__ == "__main__":
  button_sm = ButtonStateMachine("button_state_machine", 1000)
  button_sm.Start()
  while(1):
    button_sm.Execute()
