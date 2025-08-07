import os
from torch.utils.data import Dataset
from annotations import read_file

from torchvision import transforms
from PIL import Image



class PlayerActionDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.transform = transform
        self.samples = []  # list of tuples: (image_path, (x1, y1, x2, y2), label)

        for match in os.listdir(base_path):
            match_path = os.path.join(base_path, match)
            annotation_file = os.path.join(match_path, "annotations.txt")
            if not os.path.exists(annotation_file):
                continue

            video_info, _, _ = read_file(annotation_file)

            for frame_id, dic in video_info.items():
                frame_id_str = str(frame_id)
                frame_filename = f"{frame_id_str}.jpg"
                image_path = os.path.join(match_path, frame_id_str, frame_filename)

                for box in dic["boxinfos"]:
                    box_coords = (box.x1, box.y1, box.x2, box.y2)
                    label = box.activity
                    self.samples.append((image_path, box_coords, label))

        
        label_set = sorted(list(set(label for _, _, label in self.samples)))
        self.label_to_id = {label: idx for idx, label in enumerate(label_set)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, (x1, y1, x2, y2), label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        player_crop = image.crop((x1, y1, x2, y2))

        if self.transform:
            player_crop = self.transform(player_crop)

        label_id = self.label_to_id[label]
        return player_crop, label_id
