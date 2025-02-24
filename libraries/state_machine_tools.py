from typing import Callable, List
import time 

class Workable:
  is_working = False

  def Start(self):
    self.is_working = True

  def Pause(self):
    self.is_working = False

  def Reset(self):
    self.is_working = False


class PeriodManager(Workable):

  def __init__(self, period_ms: int):
    self.period_ms: int = period_ms
    self.passed_ms: int = 0

    self.old_ms: int = 0
    self.curr_ms: int = 0

  def Update(self):
    self.curr_ms = round(time.time() * 1000)
    self.passed_ms += self.curr_ms - self.old_ms
    self.old_ms = self.curr_ms



  #Interface

  #Override
  def Start(self):
    self.old_ms = round(time.time() * 1000)
    self.is_working = True

  #Override
  def Reset(self):
    self.passed_ms = 0
    self.is_working = False



  def SetPeriod(self, period_ms: int):
    self.period_ms = period_ms

  def SatisfiesPeriod(self):
    if (self.is_working):
      self.Update()
      if (self.passed_ms > self.period_ms):
        self.passed_ms = 0
        return True
      else:
        return False
    else:
      return False


class Action:

  def __init__(self, name: str, function: Callable):
    self.name = name
    self.function = function

  def __call__(self):
    self.function()



  #Interface
  def SetCallable(self, function: Callable):
    self.function = function

  def Run(self):
    self.__call__()


class ActionManager(Workable):

  def __init__(self):
    self.action_list: List[Action] = []



  #Interface
  def AddAction(self, action: Action):
    self.action_list.append(action)
  
  def RemoveAction(self, action_name: str):
    for i in self.action_list:
      if (i.name == action_name):
        self.action_list.remove(i)
        return

  def RunAllActions(self):
    if (self.is_working):
      for action in self.action_list:
        action.__call__()


class State(Workable):

  # The warning is fine
  def __init__(self, name: str, state_machine):
    self.name: str = name
    self.action_manager = ActionManager()
    self.state_machine = state_machine
    self.InitExtension()

  def InitExtension(self):
    pass



  #Interface

  #Override
  def Start(self):
    self.is_working = True
    self.action_manager.Start()

  #Override
  def Pause(self):
    self.is_working = False
    self.action_manager.Pause()

  #Override
  def Reset(self):
    self.is_working = False
    self.action_manager.Reset()


  def Execute(self):
    if (self.is_working):
        self.action_manager.RunAllActions()

  def AddStateAction(self, action: Action):
    self.action_manager.AddAction(action)

  def SetStatePeriod(self, period_ms: int):
    self.period_manager.SetPeriod(period_ms)
    
  def GetStateMachine(self):
    return self.state_machine



class StateMachine(Workable):

  def __init__(self, name: str, period_ms: int):
    self.name: str = name

    self.period_manager = PeriodManager(period_ms)
    self.state_list: List[State] = []

    self.curr_state: State = None
    self.InitExtension()

  def InitExtension(self):
    pass

  #Interface

  def Start(self): #Override
    for state in self.state_list:
      state.Start()

    self.period_manager.Start()
    self.is_working = True

  def Pause(self): #Override
    for state in self.state_list:
      state.Pause()

    self.period_manager.Pause()
    self.is_working = False

  def Reset(self): #Override
    for state in self.state_list:
      state.Reset()

    self.period_manager.Reset()
    self.is_working = False



  def AddState(self, state_name: str, state_action: Action):
    self.state_list.append(State(state_name, state_action))

  def AddState(self, state: State):
    self.state_list.append(state)

  def RemoveState(self, state_name: str):
    for state in self.state_list:
      if state.name == state_name:
        self.state_list.remove(state)
        return

  def GetState(self, state_name: str):
    for state in self.state_list:
      if state.name == state_name:
        return state

  def RunState(self, state_name: str):
    for state in self.state_list:
      if state.name == state_name:
        state.Execute()
        return

  def Execute(self):
    if (self.is_working):
      if (self.period_manager.SatisfiesPeriod()):
        for state in self.state_list:
          state.Execute()

    
