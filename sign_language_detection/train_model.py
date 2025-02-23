import sys 
sys.path.append("../")

import time

from libraries.nn_tools import *

class TrainingDataPathManager(DataPathManager):
  def InitExtension(self):
    self.AddDataPath("training_path", "./datasets/asl_alphabet/asl_alphabet_train/")
    self.AddDataPath("test_path", "./datasets/asl_alphabet/asl_alphabet_test/")
    self.AddDataPath("model_path", "./models/")
    

class NNDefaultTransform(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform
  

class NNDefaultModel(NNModel):
  def InitExtension(self):
    pass
   
   
class NNASL(NNDefault):
  def InitExtension(self):
    self.AddTrainingAttributeGroup("Default", 0.2, 32, 5, 0.001, 29)
    self.AddTrainingAttributeGroup("Speed", 0.2, 32, 5, 0.1, 29)
    self.AddTrainingAttributeGroup("Speed2", 0.2, 32, 20, 0.1, 29)
    self.AddTrainingAttributeGroup("Accurate", 0.2, 32, 10, 0.001, 29)
    self.AddTrainingAttributeGroup("Accurate2", 0.2, 1000, 100, 0.001, 29)
    self.AddTrainingAttributeGroup("Accurate3", 0.2, 256, 100, 0.001, 29)
    self.AddTrainingAttributeGroup("Accurate4", 0.2, 1024, 100, 0.001, 29)
    

    self.AddNNTransformation(NNDefaultTransform("Default"))
    self.AddNNModel(NNDefaultModel("Default"))


def GetCurrentTimeInMin():
  return round(time.time()/60)


if __name__ == "__main__":
  
  training_data_path_manager: TrainingDataPathManager = TrainingDataPathManager("training_data_path_manager")

  # Update model version per run
  model_ver: str = "1.6/"
  input_path: str = training_data_path_manager.GetLiteralDataPath("training_path")
  output_path: str = training_data_path_manager.GetLiteralDataPath("model_path") + model_ver

  nn_asl: NNASL = NNASL("nn_asl")

  timer_start: float = GetCurrentTimeInMin()
  nn_asl.Train("Accurate4", "Default", "Default", input_path, output_path)
  timer_end: float = GetCurrentTimeInMin()
  print(f"Total minutes: {timer_end-timer_start}")
  

  