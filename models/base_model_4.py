import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os 


from ..core import annotations

class LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_classes, n_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size 
        self.n_layers = n_layers       
        self.lstm = nn.LSTM(input_len, hidden_size, n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)  

    def forward(self, X):
        hidden_states = torch.zeros(self.n_layers, X.size(0), self.hidden_size, device=X.device)
        cell_states = torch.zeros(self.n_layers, X.size(0), self.hidden_size, device=X.device)

        output, hide = self.lstm(X, (hidden_states, cell_states))
        output = self.output_layer(output[:,-1,:])
        return output

# Initialize AlexNet model
model = torchvision.models.alexnet(pretrained=False) 
model.load_state_dict(torch.load('baseline2.pth'))
model.eval()  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Modify classifier to exclude the last layer
list = model.classifier.children()
model.classifier = nn.Sequential(*list[:-1])

# Define data path and transformations
path = "/kaggle/input/volleyball/volleyball_/videos"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



lstm = LSTM(input_len=4096,hidden_size=500, num_classes=8,n_layers=1)
lstm =lstm.to(device)
loss = nn.CrossEntropyLoss()   #cross entropy loss for adjusting weights 
opt= torch.optim.Adam(lstm.parameters(), lr=0.001)  #using Adam optimiser here, although other options are
                                        
# Process video data
matches = [f for f in os.listdir(path)]
l = len(matches)
limit = int(0.7 * l)

trainingmatches =matches[0:limit]



for match in trainingmatches:
    clips = [f for f in os.listdir(os.path.join(path, match))]
    clips = clips[:-1]
    annotation_file = os.path.join(path , match , "annotations.txt")
    video_info, _, _ = annotations.read_file(annotation_file)


    for clip in clips:

        
      

        framestesnors = []
        frames = sorted(f for f in os.listdir(os.path.join(path, match, clip)) if f.endswith((".jpg", ".png")))
        
        
        for frame in frames:
            fullpath = os.path.join(path, match, clip, frame)
            image = Image.open(fullpath)
            img_tensor = transform(image)
            framestesnors.append(img_tensor)
        
        
                



        batch_frames = torch.stack(framestesnors).to(device)
        with torch.no_grad():
            features = model(batch_frames)
        feature_vectors = features


    
        feature_vectors = feature_vectors.unsqueeze(0)  #keda [batchsize =1 , seqlength ,, 4096->feature vector]
        out = lstm(feature_vectors)
        id = int(clip)
        label = video_info[id]["groupactivity"]
        label = torch.tensor([label], dtype=torch.long, device=device)



        lossfunction = loss(out, label)
        opt.zero_grad()
        lossfunction.backward()
        opt.step()

#testing 
lstm.eval()

correct = 0
total = 0   
testinggmatches =matches[limit:]
with torch.no_grad():
        
    for match in testinggmatches:
        clips = [f for f in os.listdir(os.path.join(path, match))]
        clips = clips[:-1]
        annotation_file = os.path.join(path , match , "annotations.txt")
        video_info, _, _ = annotations.read_file(annotation_file)


        for clip in clips:

            
        

            framestesnors = []
            frames = sorted(f for f in os.listdir(os.path.join(path, match, clip)) if f.endswith((".jpg", ".png")))
            
            
            for frame in frames:
                fullpath = os.path.join(path, match, clip, frame)
                image = Image.open(fullpath)
                img_tensor = transform(image)
                framestesnors.append(img_tensor)
            
            
                    



            batch_frames = torch.stack(framestesnors).to(device)
            with torch.no_grad():
                features = model(batch_frames)
            feature_vectors = features


            feature_vectors = feature_vectors.unsqueeze(0) 

            out = lstm(feature_vectors)
            id = int(clip)
            label = video_info[id]["groupactivity"]
            label = torch.tensor([label], dtype=torch.long, device=device)


            predicted = out.argmax(dim=1)


            total += label.size(0)
            correct += (predicted == label).sum().item()
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
torch.save(lstm.state_dict(), "baseline4_classifier.pth")


#classifier.load_state_dict(torch.load("baseline3_classifier.pth"))



            
