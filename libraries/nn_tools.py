from typing import List
import os

import torch
import torchvision
from torchvision import transforms, datasets, models

import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import time
import random

import mediapipe as mp
import cv2
from sklearn.preprocessing import LabelEncoder

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
  
  
    
class ASLTestDataset(torch.utils.data.Dataset):
  def __init__(self, root_path, transforms=None):
    super().__init__()
    
    self.hand_detector: NNHandDetector = NNHandDetector("Default")
    self.transforms = transforms
    self.imgs = []
    self.labels = []
    self.label_encoder = LabelEncoder()
    
    # Traverse subdirectories and get images
    for label_dir in Path(root_path).iterdir():

        if label_dir.is_dir():
            print("[label_dir]: ", label_dir)
            # input()
            for img_path in label_dir.glob('*.jpg'):
                self.imgs.append(img_path)
                self.labels.append(label_dir.name)  # Using subdirectory name as label 
        else:
            print("[non_label_dir]: ", label_dir)
    self.label_encoder.fit(self.labels)
                
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, idx):
    img_path = self.imgs[idx]
    # img = Image.open(img_path).convert('RGB')
    img = self.hand_detector.RunHandDetector(img_path)
    
    label = img_path.parts[-2].split('_')[0]
    print("[img_path.parts]: ", img_path.parts)
    print("[label]: ", label)
    label = self.label_encoder.transform([label])[0]
    label = torch.tensor(label, dtype=torch.long)
    if self.transforms:
      img = self.transforms(img)
    
    return img, label
    
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
    

class NNHandDetector():
  def __init__(self, name: str):
    self.name = name

  def InitExtension(self):
    pass
  
  def RunHandDetector(self, image_path: str):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands    


    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0, min_tracking_confidence=0.5) as hands:

      img = cv2.imread(image_path)
      img = cv2.flip(img, 1)
      img_width, img_height, _ = img.shape
      print("[img.shape]: ", img.shape)

      x_min = img_width
      y_min = img_height
      x_max= 0
      y_max = 0

      results = hands.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      
      # cv2.imshow("Setup", img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

      marked_img = img.copy()
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:   
          # draw landmark       
          mp_drawing.draw_landmarks(marked_img, hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                      color=(255, 0, 0), thickness=1, circle_radius=1),
                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                      color=(125, 255, 125), thickness=1, circle_radius=1)
                                    )
          
          # update coords for cropping bounding box
          for landmark in hand_landmarks.landmark:
            curr_x = int(landmark.x * img_width)
            curr_y = int(landmark.y * img_height)
            x_min, y_min = min(curr_x, x_min), min(curr_y, y_min)
            x_max, y_max = max(curr_x, x_max), max(curr_y, y_max)
      else:
        print("[Markings]: No landmarks found")
        return cv2.resize(img, (128, 128))

      # cv2.imshow("Marked", marked_img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

      # crop image with square ratio
      padding = 25
      x_length = x_max - x_min
      y_length = y_max - y_min
      longer_side = max(x_length, y_length)
      x_padding = int((longer_side - x_length) / 2) + padding
      y_padding = int((longer_side - y_length) / 2) + padding
      x_min = max(x_min - x_padding, 0)
      y_min = max(y_min - y_padding, 0)
      x_max = min(x_max + x_padding, img_width)
      y_max = min(y_max + y_padding, img_height)

      cropped_img = marked_img[x_min:x_max, y_min:y_max]
      cropped_img = cv2.resize(cropped_img, (128, 128))
      # cv2.imshow("Cropped", img)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

      return cropped_img
    
  def RunHandDetectorImageSet(self, images):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands    

    image_set = []
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
      for img in images:
          # Desperation
          # print("Image Shape: ", img.shape)
          # print()
          img = img * 255.0
          img = img.clamp(0,255).byte()
          img = img.permute(1,2,0).cpu().numpy()
          # End desperation

          # cv2.imshow("imag  e", img)
          # cv2.waitKey(0)
          img = cv2.flip(img, 1)
          results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

          img_width, img_height, _ = img.shape
          # x_max = 0
          # y_max = 0
          # x_min = img_width
          # y_min = img_height
          
          x_max = img_width
          y_max = img_height
          x_min = 0 
          y_min = 0
          # print("x_min: ", x_min)
          # print("y_min: ", y_min)
          # print("x_max: ", x_max)
          # print("y_max: ", y_max)
          # print()
          

          marked_img = img.copy()
          if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:   
              # draw landmark       
              mp_drawing.draw_landmarks(marked_img, hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(255, 0, 0), thickness=1, circle_radius=1),
                                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                          color=(125, 255, 125), thickness=1, circle_radius=1)
                                        )
              
              # update coords for cropping bounding box
              for landmark in hand_landmarks.landmark:
                curr_x = int(landmark.x * img_width)
                curr_y = int(landmark.y * img_height)
                x_min, y_min = min(curr_x, x_min), min(curr_y, y_min)
                x_max, y_max = max(curr_x, x_max), max(curr_y, y_max)

          
          # crop image with square ratio
          padding = 25
          x_length = x_max - x_min
          y_length = y_max - y_min
          longer_side = max(x_length, y_length)
          x_padding = int((longer_side - x_length) / 2) + padding
          y_padding = int((longer_side - y_length) / 2) + padding
          x_min = max(x_min - x_padding, 0)
          y_min = max(y_min - y_padding, 0)
          x_max = min(x_max + x_padding, img_width)
          y_max = min(y_max + y_padding, img_height)

          # print("x_length: ", x_length)
          # print("y_length: ", y_length)
          # print("longer_side: ", longer_side)
          # print("x_padding: ", x_padding)
          # print("y_padding: ", y_padding)
          # print("x_min: ", x_min)
          # print("y_min: ", y_min)
          # print("x_max: ", x_max)
          # print("y_max: ", y_max)

          cropped_img = marked_img[x_min:x_max, y_min:y_max]
          # print("cropped_img shape: ", cropped_img)
          cropped_img = cv2.resize(cropped_img, (128, 128))


          cropped_img = torch.from_numpy(cropped_img).permute(2,0,1).float()
          image_set.append(cropped_img)
      return torch.stack(image_set)
    
    
    
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

    # Loadable attributes
    self.criterion = torch.nn.CrossEntropyLoss()    
    self.epoch = 0
    self.loss = None

    if (not is_mobile):
      self.model = torchvision.models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
      self.optimizer = torch.optim.Adam(self.model.parameters())
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
  

  # Model Loading
  def DefaultModel(self, model_name: str, tag_name: str):
    nn_model = self.GetModel(model_name)
    tag = self.GetTrainingAttributeGroup(tag_name)
    
    # Manual Configuration
    nn_model.epoch = 0
    nn_model.model = nn_model.model

    nn_model.in_features = nn_model.model.fc.in_features
    nn_model.optimizer = torch.optim.Adam(nn_model.model.parameters(), lr=tag.learning_rate)
    nn_model.model.fc = torch.nn.Linear(nn_model.in_features, tag.num_classes, device=self.nn_device)
    nn_model.criterion = torch.nn.CrossEntropyLoss()
    
    nn_model.model.to(self.nn_device)
    
    
  def LoadModel(self, model_name: str, model_pth_path: str, tag_name: str):

    nn_model = self.GetModel(model_name)
    tag = self.GetTrainingAttributeGroup(tag_name)
    torch.set_grad_enabled(True)

    if (not nn_model.is_mobile):
      # Standard
      nn_model.model = torch.load(model_pth_path, weights_only=False, map_location=self.nn_device)
      if (nn_model.model == None):
        print("Empty")
      model_pull = torch.load(model_pth_path, weights_only=False)

      # nn_model.epoch = model_pull["epoch"]
      nn_model.model = torchvision.models.resnet50() # 2048, 1000 default
      nn_model.model.fc = torch.nn.Linear(2048, tag.num_classes, device=self.nn_device)
      # print("--Debug--")
      # print(nn_model.model.fc)
      # print("------")
      # nn_model.model.load_state_dict(model_pull, strict=False)
      # print(f"Checksum: {torch.sum(torch.stack([p.double().abs().sum() for p in nn_model.model.parameters()]))}")
      
      preload_model_dict = nn_model.model.state_dict()
      # fc_weights = model_pull["model_state_dict"]["fc.weight"]
      # fc_weights = fc_weights[:, tag.num_classes]
      # preload_model_dict.update(fc_weights)
      nn_model.model.load_state_dict(preload_model_dict)
      nn_model.model.load_state_dict(model_pull["model_state_dict"])
      
      nn_model.model.to(self.nn_device)
      for name, layer in nn_model.model.named_children():
        layer.to(self.nn_device)
      
      # for name, param in nn_model.model.named_parameters():
      #   param.data = param.data.to(self.nn_device)
      #   if param.grad != None:
      #     param.grad = param.gard.to(self.nn_device)

      nn_model.optimizer = torch.optim.Adam(nn_model.model.parameters(), tag.learning_rate)
      nn_model.optimizer.load_state_dict(model_pull["optimizer_state_dict"])
      for state in nn_model.optimizer.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.to(self.nn_device)
      


    else:
      # Mobile : Loads model but doesn't allow continous training
      nn_model.model = torchvision.models.mobilenet_v2(pretrained=True)
      in_features = nn_model.model.classifier[1].in_features
      classes = 29
      
      nn_model.model.classifier[1] = torch.nn.Linear(in_features, classes)
      nn_model.model.load_state_dict(torch.load(model_pth_path, map_location=torch.device("cpu")))
    
    nn_model.model.to(self.nn_device)
  
  
  # Train Functions
  def ValidatePathProcessing(self, input_path: str, output_path: str, loaded_model: bool):

    input_path_exists: bool = os.path.exists(input_path)
    output_path_exists: bool = os.path.exists(output_path)
    
    print("\n----------")
    # Path verification and creation
    if not input_path_exists:
      raise FileNotFoundError(f"Input path ({input_path}) is not valid. It doesn't exist!")
    else:
      print(f"Input path: {input_path} is valid.")
    
    if not loaded_model:
      if not output_path_exists:
        print(f"Creating output path: {output_path}.")
        os.mkdir("./"+output_path)
      else:
        raise FileNotFoundError(f"Output path ({output_path}) is not valid for a new model. It already exists!")
    else:
      print(f"Continuing training on path {output_path} for previous model.")
  
  def GetPathProcessing(self, input_path: str, output_path: str, loaded_model: bool):
    print_list: List[str] = []
    input_path_exists: bool = os.path.exists(input_path)
    output_path_exists: bool = os.path.exists(output_path)
    
    print_list.append("\n----------")
    if not input_path_exists:
      print_list.append(f"Input path ({input_path}) is not valid. It doesn't exist!")
    else:
      print_list.append(f"Input path: {input_path} is valid.")
    
    if not loaded_model:
      if not output_path_exists:
        print_list.append(f"Creating output path: {output_path}.")
      else:
        print_list.append(f"Output path ({output_path}) is not valid for a new model. It already exists!")
    else:
      print_list.append(f"Continuing training on path {output_path} for previous model.")
    return print_list
    
    
  def GetTrainingAttributes(self, tag_name: str):
    tag = self.GetTrainingAttributeGroup(tag_name)
    
    return [
      "--Tag Data--", 
      f"Name: {tag.name}",     
      f"Test Size: {tag.test_size}",
      f"Batch Size: {tag.batch_size}",
      f"Num Epoch: {tag.num_epoch}",
      f"Learning Rate: {tag.learning_rate}",
      f"Num Classes: {tag.num_classes}"
    ]
  
  def GetTransformAttributes(self, transform_name: str):
    return [f"--Transform Data--", f"Name: {transform_name}"]
  
  def PrintTrainingAttributes(self, tag_name: str):
    tag_list = self.GetTrainingAttributes(tag_name)
    
    for i in tag_list:
      print(i)
    print()
    
  def PrintTransformAttributes(self, transform_name: str):
    transform_attribute_List = self.GetTransformAttributes(transform_name)

    for i in transform_attribute_List:
      print(i)
    print()
    
    
  def Train(self, tag_name: str, transform_name: str, model_name: str, input_path: str, output_path: str, worker_amount: int, loaded_model: bool, model_load_path: str, doc_path: str, model_num: str):
    
    self.ValidatePathProcessing(input_path, output_path, loaded_model)
    
    # Selection from lists
    tag = self.GetTrainingAttributeGroup(tag_name)
    nn_transformation = self.GetNNTransformation(transform_name)

    # Model Configuration
    nn_model = self.GetModel(model_name)
    if (not loaded_model):
      self.DefaultModel(model_name, tag_name)
    else:
      self.LoadModel(model_name, model_load_path, tag_name)
      
    
    # Seeding and device setting
    print()
    self.PrintTrainingAttributes(tag_name)
    self.PrintTransformAttributes(transform_name)
    
    doc_dir_path = f"./models/{model_num}/doc/"
    doc_dir_path_exists = os.path.exists(doc_dir_path)
    if (not doc_dir_path_exists):
      os.mkdir(f"./models/{model_num}/doc/")
    doc_file = open(doc_path, "a")
    doc_file.write("\n")

    path_processing_list = self.GetPathProcessing(input_path, output_path, loaded_model)
    for i in path_processing_list:
      doc_file.write(i)
      doc_file.write("\n")
    doc_file.write("\n")

      
    tag_attribute_list = self.GetTrainingAttributes(tag_name)
    for i in tag_attribute_list:
      doc_file.write(i)
      doc_file.write("\n")
    doc_file.write("\n")
      
    transform_attribute_list = self.GetTransformAttributes(transform_name)
    for i in transform_attribute_list:
      doc_file.write(i)
      doc_file.write("\n")
    doc_file.write("\n")
    

    # Import dataset and prepare
    train_dataset = ASLTestDataset(input_path, transforms=nn_transformation)
    train_dataset_size = len(train_dataset)
    indices = torch.randperm(train_dataset_size)
    split = int(train_dataset_size * tag.test_size)
    train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=tag.batch_size, shuffle=True, num_workers=worker_amount)
        
    # Training
    start_epoch: int = nn_model.epoch
    end_epoch: int = start_epoch + tag.num_epoch
    
    begin_training_list = ["\nBeginning training...\n-----------\n", 
     f"Using device: {self.nn_device}", 
     f"Starting at epoch {start_epoch}...",
     f"Reaching for {end_epoch} epochs..."]

    for i in begin_training_list:
      print(i)
      doc_file.write(i)
      doc_file.write("\n")
    doc_file.write("\n")
    doc_file.flush()

    for epoch in range(start_epoch, end_epoch):
      running_loss = 0
      correct_train = 0
      total_train = 0
      
      nn_model.model.to(self.nn_device)
      # for i in nn_model.model.parameters():
      #   print(f"Model device: ", i.device)
      
      # print(f"Model: {nn_model.model.device} ")
      # hand_detector: NNHandDetector = NNHandDetector("Default") 
      nn_model.model.train()
      for images, labels in train_dataloader:
        # images = hand_detector.RunHandDetectorImageSet(images)
        # print(len(images))
        images = torch.tensor(images)
        images = images.to(self.nn_device)
        print()
        print("[labels?tuple]: ", labels)
        print()
        labels = labels.to(self.nn_device)
        # print(f"Images device: {images.device} ")
        # print(f"Labels device: {labels.device} ")
        
        output = nn_model.model(images)
        loss = nn_model.criterion(output.to(self.nn_device), labels.to(self.nn_device))

        correct_train += (torch.max(output.to(self.nn_device), dim=1)[1] == labels.to(self.nn_device)).sum()
        total_train += labels.to(self.nn_device).size(0)

        nn_model.optimizer.zero_grad()
        loss.backward()
        nn_model.optimizer.step()

        running_loss += loss.item()

      print(f"completed epoch {epoch} with running loss {running_loss}...")
      doc_file.write(f"\ncompleted epoch {epoch} with running loss {running_loss}...")
      doc_file.flush()
      
      model_data = {
        "epoch" : epoch + 1,
        "model_state_dict" : nn_model.model.state_dict(),
        "optimizer_state_dict" : nn_model.optimizer.state_dict(),
      }
      torch.save(model_data, f'{output_path}{epoch}.pth')
   

  # Helper   
  def GetLabel(self, probabilities: List[int]):
    max: float = probabilities[0]
    max_index: int = 0
    for i in range(len(probabilities)):
      if probabilities[i] > max:
        max = probabilities[i]
        max_index = i
    return max_index
  
  def PredictNormal(self, model, image_tensor, device):
    model.eval()
    with torch.no_grad():
      image_tensor = image_tensor.to(device)
      outputs = model(image_tensor)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return self.GetLabel(probabilities.cpu().numpy().flatten())

  # Straight up gets the label
  def PredictShort(self, model_name: str, image_path: str, transform_name: str):
    nn_model = self.GetModel(model_name)
    nn_model.model.eval()
    
    nn_transform = self.GetNNTransformation(transform_name)

    hand_detector: NNHandDetector = NNHandDetector("Default") 
    input_image = Image.fromarray(hand_detector.RunHandDetector(image_path))
    image_tensor = nn_transform(input_image)[:3].unsqueeze(0)
    
    outputs = ""
    probabilities = []   
    with torch.no_grad():
      nn_model.model.eval()
      image_tensor = image_tensor.to(self.nn_device)
      outputs = nn_model.model.forward(image_tensor)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    #probabilities = probabilities.cpu().numpy().flatten()
    return self.GetLabel(probabilities.numpy().flatten())
    
  # Gets label and prints out a bunch of stuff
  def PredictLong(self, model_name: str, image_path: str, transform_name: str, enable_layer_ouput: bool, enable_classification_score: bool, enable_probability_array: bool):

    nn_model = self.GetModel(model_name)
    nn_model.model.eval()
    
    nn_transform = self.GetNNTransformation(transform_name)


    hand_detector: NNHandDetector = NNHandDetector("Default") 
    
    input_image = Image.fromarray(hand_detector.RunHandDetector(image_path))
    # print("[input_image.shape]: ", input_image.shape)
    image_tensor = nn_transform(input_image)[:3].unsqueeze(0)
    

    outputs = ""
    probabilities = []   

    with torch.no_grad():
      nn_model.model.eval()
      image_tensor = image_tensor.to(self.nn_device)
      outputs = nn_model.model.forward(image_tensor)
      # classification_score, probabilities = torch.max(outputs, 1)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
      classification_score, probability_array = torch.max(outputs, 1)
    
    probabilities = probabilities.cpu().numpy().flatten()

    if (enable_layer_ouput):
      print("\n----------\nLAYER OUTPUT\n----------")
      for name, module in nn_model.model.named_modules():
        print(f"Name: {name}, Module:{module}")
      print("\n------------------------------------\n")
      
    if (enable_classification_score):
      pass
      print(f"\nCLASSIFICATION SCORE: {classification_score.item()}")

    if (enable_probability_array):
      print(f"PROBABILITY ARRAY: \n{probabilities}")

    return self.GetLabel(probabilities)


  def RunTestMatching(self, model_name: str, transform_name: str, data_set_type: torch.utils.data.Dataset, training_data_path: str, test_data_path: str):

    nn_model = self.GetModel(model_name)
    model = nn_model.model

    device = self.nn_device

    transform = self.GetNNTransformation(transform_name)
    
    training_attribute_group: TrainingAttributeGroup = TrainingAttributeGroup("Default", 0.2, 32, 5, 0.001, 29)
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
      ax[i][j].set_title(f"P:{self.PredictNormal(model, img[:3].unsqueeze(0), device)}")
      ax[i][j].axis('off')
      j += 1
      if j == columns:
              j = 0
              i += 1
            
    plt.show()
