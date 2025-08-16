



import torchvision


from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn

from datasets.init import datasetinit
from  base_model_1 import *


from models.base_model_1 import train_test_model
from models.classifier import GroupActivityClassifier





model = torchvision.models.alexnet(pretrained=False) 
model.load_state_dict(torch.load('baseline2.pth'))
model.eval()  

list = model.classifier.children()
model.classifier =nn.Sequential(*list[:-1]) 




Data = datasetinit()
train_player_dataset = Data.train_player_dataset  
test_player_dataset = Data.test_player_dataset
train_img_level_dataset =Data.train_img_level_dataset
test_img_level_dataset =Data.test_img_level_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classifier = GroupActivityClassifier(input_dim=4096, num_classes=8)  # 8 group activities in volleyball dataset
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)




classifier = classifier.to(device)




model = model.to(device)

for i in range(0, len(train_player_dataset), 9): 
    if i + 9 > len(train_player_dataset):
        break

    optimizer.zero_grad()

    imgs = []
    for j in range(9):
        img, _ = train_player_dataset[i + j]
        imgs.append(img)
    
    
    
    imgs = torch.stack(imgs,dim=0).to(device)  

    with torch.no_grad():
        feats = model(imgs)              
    frame_feature = torch.max(feats, dim=0)[0]

    
    frame_idx = i // 9
    _, label = train_img_level_dataset[frame_idx]

    label = label.to(device)
    logits = classifier(frame_feature.unsqueeze(0))  
    loss = criterion(logits, label.unsqueeze(0))

    loss.backward()
    optimizer.step()


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i in range(0, len(test_player_dataset), 9): 
        if i + 9 > len(test_player_dataset):
            break


        imgs = []
        for j in range(9):
            img, _ = test_player_dataset[i + j]
            imgs.append(img)
    
    
    
        imgs = torch.stack(imgs,dim=0).to(device)  

        with torch.no_grad():
            feats = model(imgs)  

                    
        frame_feature = torch.max(feats, dim=0)[0]
        
        
        frame_idx = i // 9
        _, label = test_img_level_dataset[frame_idx]
        label = label.to(device)



        logits= classifier(frame_feature.unsqueeze(0))



        _, predicted = torch.max(logits, dim=1)

        total += label.size(0)
        correct += (predicted == label).sum().item()




accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
torch.save(classifier.state_dict(), "baseline3_classifier.pth")
#classifier.load_state_dict(torch.load("baseline3_classifier.pth"))


    

   

    







