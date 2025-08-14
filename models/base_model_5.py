
from lstm_classifier import LSTM
from core.annotations import read_file,BoxInfo
import os 
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision


import torch.nn as nn

path = "/kaggle/input/volleyball/volleyball_/videos"
matches = [f for f in os.listdir(path)]
l = len(matches)
limit = int(0.7 * l)
trainingmatches =matches[0:limit]



model = torchvision.models.alexnet(pretrained=False) 
model.load_state_dict(torch.load('baseline2.pth'))
model.eval()  


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

lstm = LSTM(input_len=4096,hidden_size=500, num_classes=8,n_layers=1)
lstm =lstm.to(device)
loss = nn.CrossEntropyLoss()   #cross entropy loss for adjusting weights 
opt= torch.optim.Adam(lstm.parameters(), lr=0.001)  #using Adam optimiser here, although other options are
                 

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
        for frame in middle_frames:
            cropped_imgs=[]
            for box_info in boxinfos:
                image_path = os.path.join(path , match,clip,frame)
                image = Image.open(image_path).convert('RGB')
                player_crop = image.crop((box_info.x1, box_info.y1, box_info.x2, box_info.y2))
                img_tensor = transform(player_crop)
                
                cropped_imgs.append(img_tensor)


            cropped_imgs = torch.stack(cropped_imgs, dim=0).to(device)
            with torch.no_grad():
                features = model(cropped_imgs)

            img_level_feature = torch.max(features,dim=0)[0]   #output shape: [4096]
            img_level_features.append(img_level_feature)






        img_level_features=torch.stack(img_level_features ,dim=0)  #keda el shape is [9 , 4096]
        img_level_features = img_level_features.unsqueeze(0).to(device)   #keda el shape is [1  ,  9 ,  4096]
        
        
        
        out = lstm(img_level_features)


        id = int(clip)
        label = video_info[id]["groupactivity"]
        label = torch.tensor([label], dtype=torch.long, device=device)

        lossfunction = loss(out, label)
        opt.zero_grad()
        lossfunction.backward()
        opt.step()


# ------------------ TESTING ------------------
lstm.eval()
correct = 0
total = 0
test_loss = 0.0

testingmatches = matches[limit:]
with torch.no_grad():
    for match in testingmatches:
        clips = [f for f in os.listdir(os.path.join(path, match))]
        clips = clips[:-2]  # Same skip as training
        annotation_file = os.path.join(path, match, "annotations.txt")
        video_info, _, _ = read_file(annotation_file)

        for clip in clips:
            full_path = os.path.join(path, match, clip)
            frames = sorted(f for f in os.listdir(full_path) if f.endswith((".jpg", ".png")))

            frame_indices = (16, 17, 18, 19, 20, 21, 22, 23, 24)
            middle_frames = [img for index, img in enumerate(frames) if index in frame_indices]
            clip_id = int(clip)
            boxinfos = video_info[clip_id]["boxinfos"]

            img_level_features = []
            for frame in middle_frames:
                cropped_imgs = []
                for box_info in boxinfos:
                    image_path = os.path.join(path, match, clip, frame)
                    image = Image.open(image_path).convert('RGB')
                    player_crop = image.crop((box_info.x1, box_info.y1, box_info.x2, box_info.y2))
                    img_tensor = transform(player_crop)
                    cropped_imgs.append(img_tensor)

                cropped_imgs = torch.stack(cropped_imgs, dim=0).to(device)
                features = model(cropped_imgs)

                img_level_feature = torch.max(features, dim=0)[0]  # Shape: [4096]
                img_level_features.append(img_level_feature)

            img_level_features = torch.stack(img_level_features, dim=0).unsqueeze(0).to(device)

            out = lstm(img_level_features)
            label = video_info[clip_id]["groupactivity"]
            label = torch.tensor([label], dtype=torch.long, device=device)

            loss_value = loss(out, label)
            test_loss += loss_value.item()

            predicted = out.argmax(dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # --- Logging per clip ---
            print(f"[TEST] Match {match}, Clip {clip}, Loss: {loss_value.item():.4f}, "
                  f"Predicted: {predicted.item()}, Actual: {label.item()}")

# --- Final summary ---
accuracy = 100 * correct / total
avg_loss = test_loss / total
print(f"\n[TEST SUMMARY] Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")





            
        

