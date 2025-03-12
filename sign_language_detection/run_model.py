import torch
import torchvision
from torchvision import transforms, datasets

import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import time

import sys 
sys.path.append("../")

from libraries.nn_tools import *


class RunModelPathManager(DataPathManager):
  def InitExtension(self):
    
    # Training, Testing Paths
    self.AddDataPath("training_path", "./datasets/asl_alphabet/asl_alphabet_train/")
    self.AddDataPath("test_path", "./datasets/asl_alphabet/")

    # Models Path
    self.AddDataPath("models_path", "./models/")
    
    # Singular Image Paths
    self.AddDataPath("G", "./input/G_test.jpg")
    # self.AddDataPath("i", "./hi_there_asl/i.JPG")
    # self.AddDataPath("e", "./hi_there_asl/e.JPG")


class ASLTestDataset(torch.utils.data.Dataset):
  def __init__(self, root_path, transforms=None):
    super().__init__()
    
    self.hand_detector: NNHandDetector = NNHandDetector("Default")
    self.transforms = transforms
    self.imgs = []
    self.labels = []
    self.label_encoder = LabelEncoder()
    self.known_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']
    
    # Traverse subdirectories and get images
    for label_dir in Path(root_path).iterdir():

        if label_dir.is_dir():
            for img_path in label_dir.glob('*.jpg'):
                self.imgs.append(img_path)
                self.labels.append(label_dir.name)  # Using subdirectory name as label 
        else:
            print("[non_label_dir]: ", label_dir)
    self.label_encoder.fit(self.known_labels)
                
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, idx):
    img_path = self.imgs[idx]
    img = Image.fromarray(self.hand_detector.RunHandDetector(img_path))
    
    label = img_path.parts[-1].split('_')[0]
    print("[img_path.parts]: ", img_path.parts)
    print("[label]: ", label)
    label = self.label_encoder.transform([label])[0]
    label = torch.tensor(label, dtype=torch.long)
    if self.transforms:
      img = self.transforms(img)
    
    return img, label

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
   
class NNASL(NNDefault):
  def InitExtension(self):
    self.AddTrainingAttributeGroup("Funny", 0.2, 32, 100, 0.01, 29) 

    self.AddNNTransformation(NNDefaultTransform("Default"))
    self.AddNNTransformation(NNVariableTransform("Variable"))
    self.AddNNTransformation(NNRealistic2Transform("Realistic2"))

    self.AddNNModel(NNModel("Default", False))
    self.AddNNModel(NNModel("Mobile", True))


def ConvertLabelToNum(label: int):
  return ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ',' ',' '][label]


if __name__ == "__main__":
  
  torch.manual_seed(1)

  path_manager: RunModelPathManager = RunModelPathManager("run_model_path_manager")

  tag_type: str                      = "Funny"
  transform_type: str                = "Default"
  model_type: str                    = "Default"
  model_num: str                     = "40.5"
  existing_model_num: str            = "15"
  enable_layer_output: bool          = False
  enable_classification_output: bool = True
  enable_probability_array: bool     = True
  model_path: str = path_manager.GetLiteralDataPath("models_path") + model_num + "/" + existing_model_num + ".pth"
  input_path: str = path_manager.GetLiteralDataPath("G")                                                       


  # torch.cuda.manual_seed(1)
  asl_nn_model: NNASL = NNASL("asl_nn_model")
  asl_nn_model.LoadModel(model_type, model_path, tag_type)                                                     
  
  
  # ----- Matplot Lib ------
  # Comment out if you do not want to test model on test files
  asl_nn_model.RunTestMatching(model_type, 
                               transform_type,
                               ASLTestDataset, 
                               path_manager.GetLiteralDataPath("training_path"), 
                               path_manager.GetLiteralDataPath("test_path"), 
                              )
  # ------------------------


  start_time: int = round(time.time() * 1000)
  label = asl_nn_model.PredictLong(model_type, input_path, transform_type, enable_layer_output, enable_classification_output, enable_probability_array)
  letter = ConvertLabelToNum(label)
  print(f"LABEL: {label}") 
  print(f"LETTER: {letter}") 
  end_time: int = round(time.time() * 1000)

  print(f"TOTAL TIME: {end_time-start_time} ms")