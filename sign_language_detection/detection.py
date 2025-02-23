import numpy as np
import pandas as pd
import os

import torch
import torchvision
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

 

### setting datapaths for image datasets

dataset_path = "./datasets/asl_alphabet"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

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


### preprocessing dataset

train_transforms = v2.Compose([
    v2.Resize(size=(128,128), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transforms = v2.Compose([
    v2.Resize(size=(128,128), antialias=True),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_split_percentage = 0.8

full_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)

full_dataset_size = len(full_dataset)
train_size = int(full_dataset_size * train_split_percentage)
validation_size = full_dataset_size - train_size

train_subset, validation_subset = random_split(full_dataset, [train_size, validation_size])

train_subset.dataset.transform = train_transforms
validation_subset.dataset.transform = validation_transforms


### defining parameters for training

batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 29

train_dataloader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=batch_size, shuffle=False)

classes = train_dataloader.dataset.dataset.classes

model = torchvision.models.mobilenet_v2(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


### training model

print("Starting training...")

for epoch in range(num_epochs):  
    running_loss_train = 0
    correct_train = 0
    total_train = 0

    # training loop
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss_train += loss.item()
        correct_train += (torch.max(outputs, dim=1)[1] == labels).sum()
        total_train += labels.size(0)

        progress_bar.update(1)
            
    train_accuracy = (correct_train / total_train) * 100
    avg_loss_train = running_loss_train / len(train_dataloader)

    """
    # validation loop
    model.eval()
    running_loss_validation = 0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(outputs, labels)

            running_loss_validation += loss.item()
            correct_validation += (torch.max(outputs, dim=1)[1] == labels).sum().item()
            total_validation += labels.size(0)
    
    validation_accuracy = (correct_validation / total_validation) * 100
    avg_loss_validation = running_loss_validation / len(validation_dataloader)    
    """
    
    # epoch results
    print(f"Epoch {epoch+1}/{num_epochs} completed")
    print("Avg training loss: {avg_loss_train:.4f}, Training accuracy: {train_accuracy:.2f}%")
    #print("Avg validation loss: {avg_loss_validation:.4f}, Validation accuracy: {validation_accuracy:.2f}%\n")

    # Save model after each epoch
    torch.save(model.state_dict(), f'mobilenet_asl_epoch{epoch + 1}.pth')



### final testing on test_dataset

test_dataset = datasets.ImageFolder(test_data_path, transform=validation_transforms)

columns = 7
row = round(len(test_dataset) / columns)

fig, ax = plt.subplots(row, columns, figsize=(columns * 2, row * 2))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

model.eval()
for idx, (img, label) in enumerate(test_dataset):
    img = img.to(device)
    
    prediction = model(img.unsqueeze(0))

    ax.flat[idx].imshow(img.cpu().permute(1, 2, 0))
    ax.flat[idx].set_title(f'GT {label}. Pred {classes[torch.max(prediction, dim=1)[1]]}')
    ax.flat[idx].axis('off')
        
plt.show()


