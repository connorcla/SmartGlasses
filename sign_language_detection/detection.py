import numpy as np
import pandas as pd
import os

import torch
import torchvision
from torchvision import transforms, datasets

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt



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
 

dataset_path = "./datasets/asl_alphabet"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please download it first.")

print("Dataset found at:", dataset_path)

train_data_path = './datasets/asl_alphabet/asl_alphabet_train/'
test_data_path = './datasets/asl_alphabet/asl_alphabet_test/'

"""
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training dataset not found at {train_data_path}")

if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test dataset not found at {test_data_path}")

print("Dataset paths are correct!")
"""


test_size = 0.2
batch_size = 32
num_epoch = 5
learning_rate = 0.001
num_classes = 29

train_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(128), 
    transforms.ToTensor()
])


train_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)
num_train_samples = len(train_dataset)
train_dataset

test_dataset = datasets.ImageFolder(train_data_path, transform=test_transforms)
test_dataset


torch.manual_seed(1)
indices = torch.randperm(num_train_samples)

split = int(num_train_samples * test_size)

train_dataset = torch.utils.data.Subset(train_dataset, indices[split:])
test_dataset = torch.utils.data.Subset(test_dataset, indices[:split])

len(train_dataset), len(test_dataset)

def RunTraining():
    
    training_version = input("Input training version (ex: 1.0.0): ")
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=0)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0)


    classes = train_dataloader.dataset.dataset.classes

    print("\n--------------------\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (torch.cuda.is_available()):
        print(f"Using gpus: {torch.cuda.device_count()} GPUs available...")
    else:
        print(f"Using cpu...")
    print("\n--------------------\n")
    print()

    print("\n--------------------\n")
    model = torchvision.models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    model.to(device)
    print("\n--------------------\n")
    print()

    
    print("\n--------------------")
    print("Running epochs...\n")
    for epoch in range(num_epoch):  
        running_loss = 0
        correct_train = 0
        total_train = 0

        model.train()
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)

            correct_train += (torch.max(output, dim=1)[1] == labels).sum()
            total_train += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} completed...s")


        # Save model after each epoch
        torch.save(model, f'checkpoints/{training_version}/checkpoint_{epoch + 1}.pth')
    print("\n--------------------\n")
    print("Training finished\n")

    return device, model, classes 
    

def PlotTraining(device, model, classes):
    test_data_path = Path('./datasets/asl-alphabet/asl_alphabet_test/')
    test_dataset = ASLTestDataset(test_data_path, transforms=test_transforms)

    columns = 7
    row = round(len(test_dataset) / columns)

    fig, ax = plt.subplots(row, columns, figsize=(columns * row, row * columns))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    i, j = 0, 0
    for img, label in test_dataset:
        img = torch.Tensor(img)
        img = img.to(device)
        model.eval()
        prediction = model(img[None])

        ax[i][j].imshow(img.cpu().permute(1, 2, 0))
        ax[i][j].set_title(f'GT {label}. Pred {classes[torch.max(prediction, dim=1)[1]]}')
        ax[i][j].axis('off')
        j += 1
        if j == columns:
                j = 0
                i += 1
            
    plt.show()
    return



        
if __name__ == '__main__':

    user_choice = ""
    device = model = classes = ""

    while (user_choice != "Quit"):

        print("-----\nMenu\n-----")
        print("1. Training")
        print("2. Plot\n")
        user_choice = input("Choice: ")

        if (user_choice == "Training"):
            device, model, clases = RunTraining()
        elif (user_choice == "Plot"):
            PlotTraining()
        else:
            break



