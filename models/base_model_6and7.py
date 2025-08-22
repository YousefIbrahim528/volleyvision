from core.annotations import read_file, BoxInfo
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
from models.classifier import GroupActivityClassifier, AttentivePooling
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
trainingmatches = matches[0:limit]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.alexnet(pretrained=False)
model.load_state_dict(torch.load('baseline2.pth'))
model.eval()

classifier = GroupActivityClassifier(input_dim=500, num_classes=8)  # 8 group activities in volleyball dataset
classifier = classifier.to(device)
criterion = nn.CrossEntropyLoss()

AttentivePoolingClassifier = AttentivePooling(feature_dim=4596, hidden_dim=512)
AttentivePoolingClassifier = AttentivePoolingClassifier.to(device)
criterion2 = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = model.to(device)

lstm = LSTMModel(input_dim=4096, hidden_dim=500, layer_dim=1)
lstm = lstm.to(device)

# Define two optimizers
optimizer = torch.optim.Adam(list(lstm.parameters()) + list(classifier.parameters()), lr=1e-3)
optimizer2 = torch.optim.Adam(AttentivePoolingClassifier.parameters(), lr=1e-3)

for match in trainingmatches:
    clips = [f for f in os.listdir(os.path.join(path, match))]
    clips = clips[:-2]
    annotation_file = os.path.join(path, match, "annotations.txt")
    video_info, _, _ = read_file(annotation_file)

    for clip in clips:
        full_path = os.path.join(path, match, clip)
        frames = sorted(f for f in os.listdir(full_path) if f.endswith((".jpg", ".png")))

        frame_indices = (16, 17, 18, 19, 20, 21, 22, 23, 24)
        middle_frames = [img for index, img in enumerate(frames) if index in frame_indices]
        clip_id = int(clip)
        boxinfos = video_info[clip_id]["boxinfos"]
        cropped_imgs = []

        ptis = []
        img_level_features = []
        temporal_embeddings = []
        for box_info in boxinfos:
            player_cropped_imgs = []
            for frame in middle_frames:
                image_path = os.path.join(path, match, clip, frame)
                image = Image.open(image_path).convert('RGB')
                player_crop = image.crop((box_info.x1, box_info.y1, box_info.x2, box_info.y2))
                img_tensor = transform(player_crop)
                player_cropped_imgs.append(img_tensor)

            cropped_imgs = torch.stack(player_cropped_imgs, dim=0).to(device)
            with torch.no_grad():
                player_feature_vectors = model(cropped_imgs)  # 9 imgs for the same player
                mid_frame_feature_vector = player_feature_vectors[4]

            player_feature_vectors = player_feature_vectors.unsqueeze(0).to(device)  # [1, 9, 4096]
            temporal_embedding = lstm(player_feature_vectors)
            pti = temporal_embedding + mid_frame_feature_vector  # Combining static and dynamic features
            temporal_embeddings.append(temporal_embedding.unsqueeze(0))  # [1, 500]
            ptis.append(pti)  # ptis is temporal embedding + representation of players on the pitch (500 + 4096 = 4596)

        temporal_embeddings = torch.cat(temporal_embeddings, dim=0)  # [num_players, 500]
        final_hidden_state = torch.max(temporal_embeddings, dim=0)[0]  # [500]
        ptis = torch.stack(ptis, dim=0)  # shape is [num_players, 4596]

        id = int(clip)
        label = torch.tensor([video_info[id]["groupactivity"]], dtype=torch.long, device=device)
        logits = classifier(final_hidden_state.unsqueeze(0))  # [1, 8]
        logits2 = AttentivePoolingClassifier(ptis.unsqueeze(0))

        loss = criterion(logits, label)
        loss2 = criterion2(logits2, label)

        # Zero gradients for both optimizers
        optimizer.zero_grad()
        optimizer2.zero_grad()

        # Backward pass for each loss
        loss.backward()
        loss2.backward()

        # Step for each optimizer
        optimizer.step()
        optimizer2.step()

        print(f"Match {match}, Clip {clip}, Loss: {loss.item():.4f}")
        print(f"Match {match}, Clip {clip}, Loss2: {loss2.item():.4f}")

torch.save({
    "lstm": lstm.state_dict(),
    "classifier": classifier.state_dict(),
    "attentive_pooling": AttentivePoolingClassifier.state_dict()  # Save AttentivePooling state too
}, "baseline6.pth")

# --------------------------
# Evaluation
# --------------------------
val_matches = matches[limit:]  # remaining 30% for validation
lstm.eval()
classifier.eval()
AttentivePoolingClassifier.eval()

correct6, total6 = 0, 0  # baseline6
correct7, total7 = 0, 0  # baseline7

with torch.no_grad():
    for match in val_matches:
        clips = [f for f in os.listdir(os.path.join(path, match))]
        clips = clips[:-2]
        annotation_file = os.path.join(path, match, "annotations.txt")
        video_info, _, _ = read_file(annotation_file)

        for clip in clips:
            full_path = os.path.join(path, match, clip)
            frames = sorted(f for f in os.listdir(full_path) if f.endswith((".jpg", ".png")))

            frame_indices = (16, 17, 18, 19, 20, 21, 22, 23, 24)
            middle_frames = [img for index, img in enumerate(frames) if index in frame_indices]
            clip_id = int(clip)
            boxinfos = video_info[clip_id]["boxinfos"]

            temporal_embeddings = []
            ptis = []
            for box_info in boxinfos:
                player_cropped_imgs = []
                for frame in middle_frames:
                    image_path = os.path.join(path, match, clip, frame)
                    image = Image.open(image_path).convert('RGB')
                    player_crop = image.crop((box_info.x1, box_info.y1, box_info.x2, box_info.y2))
                    img_tensor = transform(player_crop)
                    player_cropped_imgs.append(img_tensor)

                cropped_imgs = torch.stack(player_cropped_imgs, dim=0).to(device)
                with torch.no_grad():
                    player_feature_vectors = model(cropped_imgs)  # [9, 4096]
                    mid_frame_feature_vector = player_feature_vectors[4]  # static feature

                player_feature_vectors = player_feature_vectors.unsqueeze(0).to(device)  # [1, 9, 4096]
                temporal_embedding = lstm(player_feature_vectors)  # [500]
                temporal_embeddings.append(temporal_embedding.unsqueeze(0))

                pti = temporal_embedding + mid_frame_feature_vector  # [4596]
                ptis.append(pti)

            # ----- baseline6 -----
            temporal_embeddings = torch.cat(temporal_embeddings, dim=0)  # [num_players, 500]
            final_hidden_state = torch.max(temporal_embeddings, dim=0)[0]  # [500]

            label = torch.tensor([video_info[clip_id]["groupactivity"]], dtype=torch.long, device=device)
            logits6 = classifier(final_hidden_state.unsqueeze(0))  # [1, 8]
            pred6 = torch.argmax(logits6, dim=1)

            correct6 += (pred6 == label).sum().item()
            total6 += 1

            # ----- baseline7 -----
            ptis = torch.stack(ptis, dim=0)  # [num_players, 4596]
            logits7 = AttentivePoolingClassifier(ptis.unsqueeze(0))  # [1, 8]
            pred7 = torch.argmax(logits7, dim=1)

            correct7 += (pred7 == label).sum().item()
            total7 += 1

accuracy6 = 100 * correct6 / total6
accuracy7 = 100 * correct7 / total7

print(f"Validation Accuracy (Baseline6 - Max Pooling): {accuracy6:.2f}%")
print(f"Validation Accuracy (Baseline7 - Attentive Pooling): {accuracy7:.2f}%")
# This code implements two models for group activity recognition in volleyball videos.