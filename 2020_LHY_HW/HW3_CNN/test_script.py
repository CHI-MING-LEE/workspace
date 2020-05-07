import os
import numpy as np
import cv2
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        
        self.transform = transform

        if y is not None:
            self.y = torch.LongTensor(self.y)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]

        if self.transform:
            X = self.transform(X)
        
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def readfile_testset(path):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 256, 256, 3), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(256, 256))

        if i % 1000 == 0:
            print(f"{i} pics have been loaded.")
    
    return x

# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
])

# import arguments
if len(sys.argv) == 3:
    _, load_data_path, output_ans_path = sys.argv
else:
    raise ValueError("Please enter input dir and output dir!")

print("Start reading images...")
# Read dataset
test_x = readfile_testset(load_data_path)
print("Size of Testing data = {}".format(len(test_x)))

print("Make dataset and dataloader...")
# Make dataset/Make DataLoader
test_set = ImgDataset(x=test_x, transform=test_transform)
test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False)

print("Start loading model...")
# Load model
num_classes = 11
model_best = models.resnet50(pretrained=False)

# Change the last layer of model
num_ftrs = model_best.fc.in_features
model_best.fc = nn.Linear(num_ftrs, num_classes)

print("Loading trained model from git...")
# 會先用shell script load好放在local資料夾
model_best.load_state_dict(torch.load("./ResNet50_Scratch_Pre-trained_v1.model", map_location='cpu'))

print("Start predicting...")
# predict
model_best.eval()
test_label = torch.LongTensor()
with torch.no_grad():
    for inputs in test_loader:
        test_pred = model_best(inputs.cpu())
        _, ans = torch.max(test_pred, 1)

        test_label = torch.cat([test_label, ans])

prediction = test_label.numpy().tolist()

print("Wrire into csv and export...")
# 將結果寫入 csv 檔
with open(os.path.join(output_ans_path, "prediction.csv"), 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
