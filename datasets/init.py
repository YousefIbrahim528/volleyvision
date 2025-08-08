from dataset import saving_imagepath_to_txt, read_dataset_txt, ImageLevelDataset
from player_dataset import PlayerActionDataset
from torch.utils.data import random_split

class datasetinit():
    def __init__(self, transform):
        # Image-level dataset
        activity_to_id = saving_imagepath_to_txt()
        img_paths, labels = read_dataset_txt()
        length = len(img_paths)
        split_index = int(0.7 * length)

        train_paths = img_paths[:split_index]
        train_labels = labels[:split_index]
        test_paths = img_paths[split_index:]
        test_labels = labels[split_index:]

        self.train_img_level_dataset = ImageLevelDataset(train_paths, train_labels, activity_to_id, transform)
        self.test_img_level_dataset = ImageLevelDataset(test_paths, test_labels, activity_to_id, transform)

        # Player-level dataset
        path = "/kaggle/input/volleyball/volleyball_/videos"
        self.player_level_dataset = PlayerActionDataset(path)

        total_len = len(self.player_level_dataset)
        train_len = int(0.7 * total_len)
        test_len = total_len - train_len

        self.train_player_dataset, self.test_player_dataset = random_split(
            self.player_level_dataset,
            [train_len, test_len],
        )