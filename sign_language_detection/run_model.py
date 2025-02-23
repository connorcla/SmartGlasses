import numpy as np
import pandas as pd
import os

import torch
import torchvision
from torchvision import transforms, datasets

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


import sys 
sys.path.append("../")

from libraries.nn_tools import *


class RunModelPathManager(DataPathManager):
  def InitExtension(self):
    self.AddDataPath("model_ver", "./checkpoints/checkpoint_5.pth")
    self.AddDataPath("input_img", "./input-data/g_test_2.png")


transform = transforms.Compose([
  transforms.Resize(128),
  transforms.ToTensor()
])


if __name__ == "__main__":
  run_model_path_manager: RunModelPathManager = RunModelPathManager("run_model_path_manager")


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = torch.load(run_model_path_manager.GetLiteralDataPath("model_ver"), weights_only=False)
  model.eval()

  input_image = Image.open(run_model_path_manager.GetLiteralDataPath("input_img"))
  input_data = transform(input_image)[:3].unsqueeze(0)

  processed_data = ""
  with torch.no_grad():
    processed_data = model(input_data.to(device))

  output_data = torch.max(processed_data, 1)

  print(f"output_data[0]: {output_data[0].item()}")
  print(f"output_data[1]: {output_data[1].item()}")