import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import os

from datasets.init import datasetinit



def train_test_model(num_classes,traindataloader,testdataloader):
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)



    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in traindataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testdataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')



Data=datasetinit()
train_img_level_dataset =Data.train_img_level_dataset
test_img_level_dataset =Data.test_img_level_dataset
traindataloader = DataLoader(train_img_level_dataset, batch_size=32, shuffle=True)
testdataloader = DataLoader(test_img_level_dataset, batch_size=32, shuffle=True)


train_test_model(num_classes=8 , traindataloader=traindataloader,testdataloader=testdataloader)






