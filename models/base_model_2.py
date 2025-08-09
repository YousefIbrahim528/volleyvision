from base_model_1 import train_test_model
from datasets.init import datasetinit
from torch.utils.data import DataLoader, Dataset



Data = datasetinit()




train_player_dataset = Data.train_player_dataset  
test_player_dataset = Data.test_player_dataset


traindataloader = DataLoader(train_player_dataset, batch_size=32, shuffle=True)
testdataloader = DataLoader(test_player_dataset, batch_size=32, shuffle=True)

train_test_model(num_classes=9, traindataloader=traindataloader, testdataloader=testdataloader)


