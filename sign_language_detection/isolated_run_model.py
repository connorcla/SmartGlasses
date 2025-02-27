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
    self.AddDataPath("training_path", "./datasets/asl_alphabet/asl_alphabet_train/")
    self.AddDataPath("test_path", "./datasets/asl_alphabet/asl_alphabet_test/")

    self.AddDataPath("model_ver", "./checkpoints/checkpoint_5.pth")
    self.AddDataPath("model_path", "./models/")

    self.AddDataPath("g1", "./input-data/g_test.png")
    self.AddDataPath("g2", "./input-data/g_test_2.png")
    self.AddDataPath("g3", "./input-data/G_test.jpg")
    self.AddDataPath("O", "./input-data/O_test.jpg")
    self.AddDataPath("A1", "./input-data/A1.jpg")


transform = transforms.Compose([
  transforms.Resize(128),
  transforms.ToTensor()
])

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

# THIS DOESN'T WORK, PLEASE CLOSE THIS FUNCTION
def RunTestMatching(model, device, run_model_path_manager):
  training_attribute_group: TrainingAttributeGroup = TrainingAttributeGroup("Default", 0.2, 32, 5, 0.001, 29)
  train_data_path = run_model_path_manager.GetLiteralDataPath("training_path")
  torch.manual_seed(1)
  train_dataset = datasets.ImageFolder(train_data_path, transform=transform) 
  num_train_samples = len(train_dataset)
  indices = torch.randperm(num_train_samples)
  split = int(num_train_samples * 0.2)
  train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
  train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=training_attribute_group.batch_size, shuffle=False, num_workers=0)

  test_data_path = run_model_path_manager.GetLiteralDataPath("test_path")
  test_dataset = ASLTestDataset(test_data_path, transforms=transform)
  print(test_dataset.__getitem__(1)[1])

  columns = 7
  row = round(len(test_dataset) / columns)

  fig, ax = plt.subplots(row, columns, figsize=(columns * row, row * columns))
  plt.subplots_adjust(wspace=0.1, hspace=0.2)

  i, j = 0, 0
  for img, label in test_dataset:
    img = torch.Tensor(img)
    img = img.to(device)
    prediction = model.forward(img[None])

    print(f"Predicition: {prediction}")
    ax[i][j].imshow(img.cpu().permute(1, 2, 0))
    ax[i][j].set_title(f"GT {label}. Pred{Predict(model, img, device)}")
    ax[i][j].axis('off')
    j += 1
    if j == columns:
            j = 0
            i += 1
          
  plt.show()

  
def Predict(model, image_tensor, device):
  model.eval()
  with torch.no_grad():
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
  return probabilities.cpu().numpy().flatten()

def DetectLabel(array_classes: List[int]):
  for i in range(len(array_classes)):
    if i == 1:
      return i
  print("Invalid Label Generated")
  return -1
    
    
if __name__ == "__main__":
  
  run_model_path_manager: RunModelPathManager = RunModelPathManager("run_model_path_manager")

  # Make sure model version is correct
  # Unlike training model, we specify the last digit in model version
  model_ver: str = "34.0/26.pth"
  model_path: str = run_model_path_manager.GetLiteralDataPath("model_path") + model_ver
  input_path: str = run_model_path_manager.GetLiteralDataPath("g3")

  dict = torch.load(model_path, weights_only=False)
  model = torchvision.models.resnet50(pretrained=False)


  device_a = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.fc = torch.nn.Linear(2048, 29, device=device_a)
  model.load_state_dict(dict["model_state_dict"]) 
  model.to(device_a)
  model.eval()


  # UNCOMMENT TO SEE PLOT ON TEST DATASET : DOESNT WORK
#   RunTestMatching(model, device_a, run_model_path_manager)


  # UNCOMMENT TO SEE LAYERS
  # for name, module in model.named_modules():
  #   print(f"Name: {name}, Module:{module}")
  
  print("\n-----")
  print(f"Using model: {model_ver}")
  print(f"Output for {input_path}: ")

  start_time: int = round(time.time() * 1000)
  input_image = Image.open(input_path)
  input_data = transform(input_image)[:3].unsqueeze(0)


  # UNCOMMENT IF YOU WANT TO SEE CLASSIFICATION SCORE AND BOUNDING BOX STUFF
  processed_data = ""
  with torch.no_grad():
    model.eval()
    processed_data = model.forward(input_data.to(device_a))
    # processed_data = model(input_data)

  probability, label = torch.max(processed_data, 1)
  print()
  print(f"classification score: {probability.item()}")
  print(f"bounding box: {label.item()}")
  

  predict = Predict(model, input_data, device_a)
  end_time: int = round(time.time() * 1000)

  print(f"Classes Array: {predict}")
  print(f"Label: {DetectLabel(predict)}")
  print(f"Total time: {end_time-start_time} ms")