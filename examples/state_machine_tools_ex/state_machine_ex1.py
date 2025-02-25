import sys 
sys.path.append("../..")

from libraries.state_machine_tools import *



class ButtonStateMachine(StateMachine):
  def InitExtension(self): 
    self.AddState(InitializeState("initialize_state", self))
    self.AddState(OffState("off_state", self))
    self.AddState(OnState("on_state", self))
    
    # Declare state machine variables with "self.[variable_name]"
    self.button_statement = "From the state-machine!"
    
    # Declaring starting state
    self.curr_state: State = self.GetState("initialize_state")

  # Made a helper function to access state machine variable
  # Not neccessary, like most python variables you can directly access variable but I find this nicer
  # Note that python will not be able to detect dynamically created variables and functions so intellisense
  # won't be that useful these types
  def GetButtonStatement(self):
    return self.button_statement
  
  # Main execution
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
    # Declare extra state variables with "self.[variable_name]"
    
    # Add state actions, functions that run when the state machine is called
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

    # Notice that intellisense marks the GetButtonStatement() function white, it doesn't exist statically
    print(self.GetStateMachine().GetButtonStatement())      




if __name__ == "__main__":
  button_sm = ButtonStateMachine("button_state_machine", 1000)
  button_sm.Start()
  while(1):
    button_sm.Execute()
