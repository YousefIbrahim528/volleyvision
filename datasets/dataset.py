from cProfile import label
from operator import length_hint
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os
from annotations import read_file

from torch.utils.data import DataLoader

def saving_imagepath_to_txt(save_path="data/splits/dataset.txt"):
    base_path = "/kaggle/input/volleyball/volleyball_/videos"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    label_set=set()
    with open(save_path, "w") as f_out:
        for match in os.listdir(base_path):  # match = 0, 1, 2, ...
            annotation_file = os.path.join(base_path, match, "annotations.txt")
            if not os.path.exists(annotation_file):
                continue

            video_info, _, _ = read_file(annotation_file)

            for frame_id, dic in video_info.items():
                frame_filename = f"{frame_id}.jpg"
                rel_path = os.path.join(match, str(frame_id), frame_filename)
                label = dic["groupactivity"]
                label_set.add(label)  
                f_out.write(f"{rel_path} {label}\n")

    activity_list = sorted(list(label_set))  
    activity_to_id = {activity: idx for idx, activity in enumerate(activity_list)}
    return activity_to_id
        


def read_dataset_txt(path="data/splits/dataset.txt", base_path="/kaggle/input/volleyball/volleyball_/videos"):
    img_paths = []
    labels = []

    with open(path, "r") as file:
        for line in file:
            rel_path, label = line.strip().split()
            full_path = os.path.join(base_path, rel_path)
            img_paths.append(full_path)
            labels.append(label)

    return img_paths, labels




class ImageLevelDataset(Dataset):
    def __init__(self, image_paths, labels, label_map, transform=None):
        self.image_paths = image_paths
        self.labels = [label_map[label] for label in labels]  # Convert to int
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



transform = transforms.Compose([
    transforms.Resize(256),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])




# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32)