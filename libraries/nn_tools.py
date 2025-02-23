from typing import List
import os

import torch
import torchvision
from torchvision import transforms, datasets

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
  
  # VERY CONSTRUCTOR DEPENDENT
  def __init__(self, name: str):
    self.name: str = name
    self.model = torchvision.models.resnet50(weights=True)
    
    for param in self.model.parameters():
      param.requires_grad = False

    self.InitExtension()
    
  def InitExtension():
     pass
  

class NNDefault():
  def __init__(self, name: str):
    self.name: str = name
    
    self.training_attribute_group_collection: List[TrainingAttributeGroup] = []
    self.nn_transform_collection: List[NNTransform] = []
    self.nn_model_collection: List[NNModel] = []

    self.nn_device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
  
  # def AddTrainingAttributeGroup(self, training_attribute_group: TrainingAttributeGroup):
  #   self.training_attribute_group_collection.append(training_attribute_group)

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
      

  # Train
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model.model.to(device)

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
        images = images.to(device)
        labels = labels.to(device)
        
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
  