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
    random_rotation = random.uniform(0, 50)
    random_brightness = random.uniform(0, 2.0)
    random_contrast = random.uniform(0, 2.0)
    random_saturation = random.uniform(0, 2.0)
    random_hue = random.uniform(0, 0.5)
    transform = transforms.Compose([
      transforms.RandomRotation(random_rotation),
      transforms.ColorJitter(brightness=random_brightness, 
                             contrast=random_contrast, 
                             saturation=random_saturation, 
                             hue=random_hue),
      transforms.RandomPerspective(0.5, 0.1),
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform

class NNRealisticTransform(NNTransform):
    def GetTransformation(self):
      random_rotation = random.uniform(0, 40)
      random_brightness = random.uniform(0.7, 1.5)
      random_contrast = random.uniform(0.7, 1.5)
      random_saturation = random.uniform(0.7, 1.5)
      random_hue = random.uniform(0, 0.5)
      transform = transforms.Compose([
        transforms.RandomRotation(random_rotation),
        transforms.ColorJitter(brightness=random_brightness, 
                              contrast=random_contrast, 
                              saturation=random_saturation, 
                              hue=random_hue),
        transforms.RandomPerspective(0.5, 0.2),
        transforms.RandomCrop(size=(128,128)),
        transforms.Resize(128),
        transforms.ToTensor()
      ])
      return transform
    
class NNRealistic2Transform(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.RandomRotation(degrees=(0,45)),
      transforms.ColorJitter(brightness=0.5, 
                            contrast=0.3, 
                            saturation=0.5, 
                            hue=0.4),
      transforms.RandomPerspective(0.5, 0.2),
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform 
  
class NNRealistic3Transform(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.RandomRotation(degrees=(-10,10)),
      transforms.ColorJitter(brightness=0.2, 
                            contrast=0.2, 
                            saturation=0.1, 
                            hue=0.1),
      transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform
class NNRealistic3TransformHigh(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.RandomRotation(degrees=(-10,10)),
      transforms.ColorJitter(brightness=0.2, 
                            contrast=0.2, 
                            saturation=0.1, 
                            hue=0.1),
      transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
      transforms.Resize(224),
      transforms.ToTensor()
    ])
    return transform
   
class NNASL(NNDefault):
  def InitExtension(self):
    self.AddTrainingAttributeGroup("Default", 0.2, 32, 5, 0.001, 29)
    self.AddTrainingAttributeGroup("Speed", 0.2, 32, 100, 0.01, 29)
    self.AddTrainingAttributeGroup("Accurate", 0.20, 256, 500, 0.01, 29)
    self.AddTrainingAttributeGroup("Funny", 0.2, 32, 200, 0.01, 29)
    
    self.AddNNTransformation(NNDefaultTransform("Default"))
    self.AddNNTransformation(NNVariableTransform("Variable"))
    self.AddNNTransformation(NNRealisticTransform("Realistic"))
    self.AddNNTransformation(NNRealistic2Transform("Realistic2"))
    self.AddNNTransformation(NNRealistic3Transform("Realistic3"))
    self.AddNNTransformation(NNRealistic3TransformHigh("Realistic3High"))

    self.AddNNModel(NNModel("Default", False))
    self.AddNNModel(NNModel("Mobile", True))


def GetCurrentTimeInMin():
  return round(time.time()/60)


if __name__ == "__main__":
  
  path_manager: TrainingDataPathManager = TrainingDataPathManager("path_manager")

  tag_type: str                 = "Funny"
  transform_type: str           = "Realistic3High"
  model_type: str               = "Default"
  model_num: str                = "50.3"
  existing_model_num: str       = "0"
  input_path: str               = path_manager.GetLiteralDataPath("training_path")
  output_path: str              = path_manager.GetLiteralDataPath("model_path") + model_num + "/"
  num_workers: int              = 16
  is_model_loaded: bool         = False
  load_existing_model_path: str = path_manager.GetLiteralDataPath("model_path") + model_num + "/" + existing_model_num + ".pth"  
  documentation_path: str       = path_manager.GetLiteralDataPath("model_path") + model_num + "/doc/" + model_num + "." + existing_model_num + ".txt"

  torch.manual_seed(1)
  nn_asl: NNASL = NNASL("nn_asl")
    
  timer_start: float = GetCurrentTimeInMin()
  nn_asl.Train(tag_type, transform_type, model_type, input_path, output_path, num_workers, is_model_loaded, load_existing_model_path, documentation_path, model_num)       
  timer_end: float = GetCurrentTimeInMin()

  print(f"Total minutes: {timer_end-timer_start}")
  
  