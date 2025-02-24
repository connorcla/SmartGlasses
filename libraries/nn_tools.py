from typing import List
import os

import torch
import torchvision
from torchvision import transforms, datasets

import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import time

class DataPath():
  def __init__(self, path_name: str, literal_path: str):
    self.path_name = path_name
    self.literal_path = literal_path

    self.InitExtension()

  def InitExtension(self):
    pass





class DataPathManager():
  def __init__(self, path_manager_name: str):
    self.path_manager_name = path_manager_name

    self.data_paths_list: List[DataPath] = []
    self.InitExtension()
    
  def InitExtension(self):
    pass
    
    
    
  def AddDataPath(self, path_name: str, literal_path: str):
    if not os.path.exists(literal_path):
      raise FileNotFoundError(f"Path ({literal_path}) is not valid.")
    self.data_paths_list.append(DataPath(path_name, literal_path))
    print(f"Added \'{path_name}\': {literal_path}")

  def RemoveDataPath(self, path_name):
    for i in self.data_paths_list:
      if (i.path_name == path_name):
        self.data_paths_list.remove(path_name)
        return
  
  def GetLiteralDataPath(self, path_name: str):
    for i in self.data_paths_list:
      if (i.path_name == path_name):
        return i.literal_path
    raise LookupError(f"\'{path_name}\' was not found in the data manager.")
  
  
    
    
    
class TrainingAttributeGroup():
  # VERY CONSTRUCTOR DEPENDENT
  def __init__(self, name: str, test_size: float, batch_size: int, num_epoch: int, learning_rate: float, num_classes: int):
    self.name = name
    self.test_size = test_size
    self.batch_size = batch_size
    self.num_epoch = num_epoch
    self.learning_rate = learning_rate
    self.num_classes = num_classes
    self.InitExtension()
  
  def InitExtension(self):
    pass
    
    
    
    
    
class NNTransform():
  def __init__(self, name: str):
    self.name = name
    self.InitExtension()

  def InitExtension(self):
    pass
  
  # REQUIRE OVERRIDE
  def GetTransformation(self) -> transforms.Compose:
    pass





class NNModel():
  
  def __init__(self, name: str, is_mobile: bool):
    self.name: str = name
    self.is_mobile: bool = is_mobile

    if (is_mobile):
      self.model = torchvision.models.resnet50(weights=True)
    else:
      self.model = torchvision.models.mobilenet_v2(pretrained=True)
    
    for param in self.model.parameters():
      param.requires_grad = False

    self.InitExtension()
    
  def InitExtension(self):
    pass
  
  
  
  

class NNDefault():
  def __init__(self, name: str):

    self.name: str = name

    self.nn_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.training_attribute_group_collection: List[TrainingAttributeGroup] = []
    self.nn_transform_collection: List[NNTransform] = []
    self.nn_model_collection: List[NNModel] = []

    self.InitExtension()
  
  def InitExtension():
    pass
  
  
  
  # Model
  def AddNNModel(self, nn_model: NNModel):
    self.nn_model_collection.append(nn_model)
  
  def RemoveNNModel(self, nn_model_name: str):
    for i in self.nn_model_collection:
      if (i.name == nn_model_name):
        self.nn_model_collection.remove(i)
        return
  
  def GetModel(self, nn_model_name: str) -> NNModel:
    for i in self.nn_model_collection:
      if (i.name == nn_model_name):
        return i
  
  
  
  # Training Attribute Groups
  def AddTrainingAttributeGroup(self, name: str, test_size: float, batch_size: int, num_epoch: int, learning_rate: float, num_classes: int):
    self.training_attribute_group_collection.append(TrainingAttributeGroup(name, test_size, batch_size, num_epoch, learning_rate, num_classes))
  
  def RemoveTraningAttributeGroup(self, training_attribute_group_name: str):
    for i in self.training_attribute_group_collection:
      if (i.name == training_attribute_group_name):
        self.training_attribute_group_collection.remove(i)
        return
  
  def GetTrainingAttributeGroup(self, training_attribute_group_name: str):
    for i in self.training_attribute_group_collection:
      if (i.name == training_attribute_group_name):
        return i



  # NN Transformation
  def AddNNTransformation(self, nn_transformation: NNTransform):
    self.nn_transform_collection.append(nn_transformation)

  def RemoveNNTransformation(self, nn_transformation_name: str):
    for i in nn_transformation_name:
      if (i.name == nn_transformation_name):
        self.nn_transform_collection.remove(i)
        return
      
  def GetNNTransformation(self, nn_transformation_name: str) -> transforms.Compose:
    for i in self.nn_transform_collection:
      if (i.name == nn_transformation_name):
        return i.GetTransformation()
  


  # Load
  def LoadModel(self, model_name: str, model_pth_path: str):

    nn_model = self.GetModel(model_name)

    if (not nn_model.is_mobile):
      # Standard
      nn_model.model = torch.load(model_pth_path, weights_only=False)
    else:
      # Mobile
      nn_model.model = torchvision.models.mobilenet_v2(pretrained=True)
      in_features = nn_model.model.classifier[1].in_features
      classes = 29
      
      nn_model.model.classifier[1] = torch.nn.Linear(in_features, classes)
      nn_model.model.load_state_dict(torch.load(model_pth_path, map_location=torch.device("cpu")))
    
    nn_model.model.to(self.nn_device)
  
  
  
  # Train Functions
  
  def Train(self, training_attribute_group_name: str, nn_transformation_name: str, model_name: str, input_path: str, output_path: str):
    
    # Path verification and creation
    if not os.path.exists(input_path):
      raise FileNotFoundError(f"Input path ({input_path}) is not valid. It doesn't exist!")
    else:
      print(f"Input path: {input_path} is valid.")
    
    if not os.path.exists(output_path):
      print(f"Creating output path: {output_path}.")
      os.mkdir("./"+output_path)
    else:
      raise FileNotFoundError(f"Output path ({output_path}) is not valid. It already exists!")
    
    # Selection from lists
    training_attribute_group = self.GetTrainingAttributeGroup(training_attribute_group_name)
    nn_transformation = self.GetNNTransformation(nn_transformation_name)

    # Model Configuration
    nn_model = self.GetModel(model_name)
    nn_model.in_features = nn_model.model.fc.in_features
    nn_model.model.fc = torch.nn.Linear(nn_model.in_features, training_attribute_group.num_classes)
    nn_model.criterion = torch.nn.CrossEntropyLoss()
    nn_model.optimizer = torch.optim.Adam(nn_model.model.parameters(), lr=training_attribute_group.learning_rate)

    # Seeding and device setting
    torch.manual_seed(1)
    nn_model.model.to(self.nn_device)

    # Import dataset and prepare
    train_dataset = datasets.ImageFolder(input_path, transform=nn_transformation)
    train_dataset_size = len(train_dataset)
    indices = torch.randperm(train_dataset_size)
    split = int(train_dataset_size * training_attribute_group.test_size)
    train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=training_attribute_group.batch_size, shuffle=True, num_workers=10)
        
    # Training
    print("\nBeginning training...\n-----------\n")
    print(f"Reaching for {training_attribute_group.num_epoch} epochs...")
    for epoch in range(training_attribute_group.num_epoch):
      running_loss = 0
      correct_train = 0
      total_train = 0

      nn_model.model.train()
      for images, labels in train_dataloader:
        images = images.to(self.nn_device)
        labels = labels.to(self.nn_device)
        
        output = nn_model.model(images)
        loss = nn_model.criterion(output, labels)

        correct_train += (torch.max(output, dim=1)[1] == labels).sum()
        total_train += labels.size(0)

        nn_model.optimizer.zero_grad()
        loss.backward()
        nn_model.optimizer.step()

        running_loss += loss.item()

      print(f"completed epoch {epoch} with running loss {running_loss}...")
      torch.save(nn_model.model, f'{output_path}{epoch}.pth')
   


  # Helper   
  def GetLabel(self, probabilities: List[int]):
    max: float = probabilities[0]
    max_index: int = 0
    for i in range(len(probabilities)):
      if probabilities[i] > max:
        max = probabilities[i]
        max_index = i
    return max_index
  
  # Straight up gets the label
  def PredictShort(self, model, image_tensor, device):
    model.eval()
    with torch.no_grad():
      image_tensor = image_tensor.to(device)
      outputs = model(image_tensor)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return self.GetLabel(probabilities.cpu().numpy().flatten())
    

  # Gets label and prints out a bunch of stuff
  def PredictLong(self, model_name: str, image_path: str, enable_layer_ouput: bool, enable_classification_score: bool, enable_probability_array: bool):

    nn_model = self.GetModel(model_name)
    nn_model.model.eval()
    
    nn_transform = self.GetNNTransformation("Default")

    input_image = Image.open(image_path)
    image_tensor = nn_transform(input_image)[:3].unsqueeze(0)
    
    outputs = ""
    probabilities = []   
    with torch.no_grad():
      image_tensor = image_tensor.to(self.nn_device)
      outputs = nn_model.model(image_tensor)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
      classification_score, probability_array = torch.max(outputs, 1)
    
    probabilities = probabilities.cpu().numpy().flatten()

    if (enable_layer_ouput):
      print("\n----------\nLAYER OUTPUT\n----------")
      for name, module in nn_model.model.named_modules():
        print(f"Name: {name}, Module:{module}")
      print("\n------------------------------------\n")
      
    if (enable_classification_score):
      print(f"\nCLASSIFICATION SCORE: {classification_score.item()}")

    if (enable_probability_array):
      print(f"PROBABILITY ARRAY: \n{probabilities}")

    return self.GetLabel(probabilities)

  def RunTestMatching(self, model_name: str, data_set_type: torch.utils.data.Dataset, training_data_path: str, test_data_path: str):

    nn_model = self.GetModel(model_name)
    model = nn_model.model

    device = self.nn_device

    transform = self.GetNNTransformation("Default")
    
    training_attribute_group: TrainingAttributeGroup = TrainingAttributeGroup("Default", 0.2, 32, 5, 0.001, 29)
    torch.manual_seed(1)
    train_dataset = datasets.ImageFolder(training_data_path, transform=transform) 
    num_train_samples = len(train_dataset)
    indices = torch.randperm(num_train_samples)
    split = int(num_train_samples * 0.2)
    train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=training_attribute_group.batch_size, shuffle=False, num_workers=0)

    test_dataset = data_set_type(test_data_path, transforms=transform)
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

      # print(f"Predicition: {prediction}")
      ax[i][j].imshow(img.cpu().permute(1, 2, 0))
      ax[i][j].set_title(f"P:{self.PredictShort(model, img[:3].unsqueeze(0), device)}")
      ax[i][j].axis('off')
      j += 1
      if j == columns:
              j = 0
              i += 1
            
    plt.show()