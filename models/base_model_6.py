from core.annotations import read_file,BoxInfo
import os 
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
from models.classifier import GroupActivityClassifier

import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        _, (hn, cn) = self.lstm(x, (h0, c0))
        return hn.squeeze(0)  # [hidden_dim]

path = "/kaggle/input/volleyball/volleyball_/videos"
matches = [f for f in os.listdir(path)]
l = len(matches)
limit = int(0.7 * l)
trainingmatches =matches[0:limit]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.alexnet(pretrained=False) 
model.load_state_dict(torch.load('baseline2.pth'))
model.eval()  



classifier = GroupActivityClassifier(input_dim=500, num_classes=8)  # 8 group activities in volleyball dataset
criterion = nn.CrossEntropyLoss()




classifier = classifier.to(device)





transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



model = model.to(device)

lstm = LSTMModel(input_dim=4096, hidden_dim=500, layer_dim=1, output_dim=500)
lstm =lstm.to(device)

optimizer = torch.optim.Adam(
    list(lstm.parameters()) + list(classifier.parameters()), lr=1e-3
)
                 

for match in trainingmatches:
    clips = [f for f in os.listdir(os.path.join(path, match))]
    clips = clips[:-2]
    annotation_file = os.path.join(path , match , "annotations.txt")
    video_info, _, _ = read_file(annotation_file)
    

    for clip in clips:
        full_path = os.path.join(path, match, clip)
        frames = sorted(f for f in os.listdir(full_path) if f.endswith((".jpg", ".png")))

        frame_indixes= (16,17,18,19,20,21,22,23,24)
        middle_frames = [img for index, img in enumerate(frames) if index in frame_indixes ]
        clip_id = int(clip)
        boxinfos = video_info[clip_id]["boxinfos"]
        cropped_imgs=[]

        # player_crop = image.crop((x1, y1, x2, y2))

        img_level_features = []
        temporal_embeddings = []
        for box_info in boxinfos:
            player_cropped_imgs=[]
            for frame in middle_frames:

                image_path = os.path.join(path , match,clip,frame)
                image = Image.open(image_path).convert('RGB')
                player_crop = image.crop((box_info.x1, box_info.y1, box_info.x2, box_info.y2))
                img_tensor = transform(player_crop)
                
                player_cropped_imgs.append(img_tensor) #appending 9 tensors for the same player
            
            
            cropped_imgs = torch.stack(player_cropped_imgs, dim=0).to(device)  
            with torch.no_grad():
                player_feature_vectors = model(cropped_imgs)
            player_feature_vectors=player_feature_vectors.unsqueeze(0).to(device)   ##  [1, 9 ,4k]  
            temporal_embedding = lstm(player_feature_vectors)

            temporal_embeddings.append(temporal_embedding.unsqueeze(0))  # [1, 500]


        #by the end of this for loop u have hiddens for each player on the pitch 
        ##[9 hiddenstates==> each 1 represents a sequence in 9 frames   ,    size of hidden ==>500]
        temporal_embeddings = torch.cat(temporal_embeddings, dim=0)  # [num_players, 500]
        final_hidden_state = torch.max(temporal_embeddings, dim=0)[0]  # [500]

        #pass final_hidden_state embedding to a neural network  to classify the group activity 


        id = int(clip)
        label = video_info[id]["groupactivity"]
        label = torch.tensor([video_info[id]["groupactivity"]], dtype=torch.long, device=device)
        logits = classifier(final_hidden_state.unsqueeze(0))   # [1, 8]
        loss = criterion(logits, label)  # both [1]


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Match {match}, Clip {clip}, Loss: {loss.item():.4f}")

torch.save({
    "lstm": lstm.state_dict(),
    "classifier": classifier.state_dict()
}, "baseline6.pth")




                