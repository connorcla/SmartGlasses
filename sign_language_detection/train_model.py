import sys 
sys.path.append("../")

import time

from libraries.nn_tools import *

class TrainingDataPathManager(DataPathManager):
  def InitExtension(self):
    
    # Must create these paths
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
class NNVariableTransform(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.RandomRotation(30),
      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform

   
class NNASL(NNDefault):
  def InitExtension(self):
    self.AddTrainingAttributeGroup("Default", 0.2, 32, 5, 0.001, 29)
    self.AddTrainingAttributeGroup("Speed", 0.2, 32, 100, 0.01, 29)
    self.AddTrainingAttributeGroup("Accurate", 0.2, 32, 100, 0.001, 29)
    self.AddTrainingAttributeGroup("Funny", 0.2, 32, 100, 0.0001, 29)
    
    self.AddNNTransformation(NNDefaultTransform("Default"))
    self.AddNNTransformation(NNVariableTransform("Variable"))

    self.AddNNModel(NNModel("Default", False))
    self.AddNNModel(NNModel("Mobile", True))


def GetCurrentTimeInMin():
  return round(time.time()/60)


if __name__ == "__main__":
  
  path_manager: TrainingDataPathManager = TrainingDataPathManager("path_manager")

  load_model_option: bool = True                                                                 # Want to load an existing model before training?

  input_path: str = path_manager.GetLiteralDataPath("training_path")
  output_path: str = path_manager.GetLiteralDataPath("model_path") + "10.0/"                     # Edit last string to be a nonexisting path in "models/"
  load_existing_model_path: str = path_manager.GetLiteralDataPath("model_path") + "10.0/0.pth"   # If load_model_option: Edit last string to be an existing path to a .pth

  nn_asl: NNASL = NNASL("nn_asl")
  if (load_model_option):
    nn_asl.LoadModel("Default", load_existing_model_path)                                        # Edit model type if needed: "Default", "Mobile"
    
  timer_start: float = GetCurrentTimeInMin()
  
  # Edit training type, transformation type, model type, input path, output path, number of workers, using a previous model
  nn_asl.Train("Funny", "Variable", "Default", input_path, output_path, 10, load_model_option)       
  timer_end: float = GetCurrentTimeInMin()

  print(f"Total minutes: {timer_end-timer_start}")
  

  