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
    self.AddDataPath("test_path", "./datasets/asl_alphabet/asl_alphabet_test/")

    # Models Path
    self.AddDataPath("models_path", "./models/")
    
    # Singular Image Paths
    self.AddDataPath("G", "./input/G_test.jpg")


class ASLTestDataset(torch.utils.data.Dataset):
  def __init__(self, root_path, transforms=None):
    super().__init__()
    
    self.transforms = transforms
    self.imgs = sorted(list(Path(root_path).glob('*.jpg')))
      
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, idx):
    img_path = self.imgs[idx]
    img = Image.open(img_path).convert('RGB')
    
    label = img_path.parts[-1].split('_')[0]
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

   
class NNASL(NNDefault):
  def InitExtension(self):
    self.AddNNTransformation(NNDefaultTransform("Default"))
    self.AddNNTransformation(NNVariableTransform("Variable"))

    self.AddNNModel(NNModel("Default", False))
    self.AddNNModel(NNModel("Mobile", True))


def ConvertLabelToNum(label: int):
  return ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ',' ',' '][label]


if __name__ == "__main__":
  
  path_manager: RunModelPathManager = RunModelPathManager("run_model_path_manager")

  model_path: str = path_manager.GetLiteralDataPath("models_path") + "6.0/53.pth"       # Specify model path name here
  input_path: str = path_manager.GetLiteralDataPath("G")                                # Specify input path name here

  asl_nn_model: NNASL = NNASL("asl_nn_model")
  asl_nn_model.LoadModel("Default", model_path)                                         # Change to "Modile" if needed
  
  
  # ----- Matplot Lib ------
  # Comment out if you do not want to test model on test files
  # asl_nn_model.RunTestMatching("Default", 
  #                              ASLTestDataset, 
  #                              path_manager.GetLiteralDataPath("training_path"), 
  #                              path_manager.GetLiteralDataPath("test_path"), 
  #                             )
  # ------------------------


  start_time: int = round(time.time() * 1000)
  label = asl_nn_model.PredictLong("Default", input_path, False, True, True)            # Customize prediction parameters
  letter = ConvertLabelToNum(label)
  print(f"LABEL: {label}") 
  print(f"LETTER: {letter}") 
  end_time: int = round(time.time() * 1000)

  print(f"TOTAL TIME: {end_time-start_time} ms")