#%% import libraries
import os
import cv2
from PIL import Image
import atom
import atom.data_cleaning as dc
import atom.feature_engineering as fe
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
path = 'data/images'
pixels_per_side = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)
#%% building the dataset
data = []
labels = []
folder = os.listdir(path)
for file in folder:
    img = cv2.imread(str(os.path.join(path, file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (pixels_per_side, pixels_per_side))
    data.append(img)
    labels.append(int(file[:3]) - 1)
labels = np.array(labels)
#labels = labels.reshape(-1, 1)
label_dict = {0: 'Danaus plexippus',
              1: 'Heliconius charitonius',
              2: 'Heliconius erato',
              3: 'Junonia coenia',
              4: 'Lycaena phlaeas',
              5: 'Nymphalis antiopa',
              6: 'Papilio cresphontes',
              7: 'Pieris rapae',
              8: 'Vanessa atalanta',
              9: 'Vanessa cardui'}
print([label_dict[labels[i]] for i in range(10)])
#%%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = Image.fromarray(data, mode='RGB')
        if self.transform:
            data = self.transform(data)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, labels
#%%
trans = transforms.Compose([
    transforms.ToTensor()
])
dataset = MyDataset(data, labels, transform=trans)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
total_sum = torch.zeros(3)
total_count = 0
for images, _ in dataloader:
    total_sum += images.sum(dim=[0, 2, 3])
    total_count += images.numel() / images.shape[1]
mean = total_sum / total_count
sum_of_squared_diff = torch.zeros(3)
for images, _ in dataloader:
    sum_of_squared_diff += ((images - mean.unsqueeze(1).unsqueeze(2))**2).sum(dim=[0, 2, 3])
std = torch.sqrt(sum_of_squared_diff / total_count)
mean = [mean[0].item(), mean[1].item(), mean[2].item()]
std = [std[0].item(), std[1].item(), std[2].item()]
print(mean, std)
#%%
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
dataset = MyDataset(data, labels, transform=trans)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
data = []
for img, _ in dataloader:
    data.append(img.numpy())
data = np.concatenate(data, axis=0).transpose((0, 2, 3, 1))
#%%
print(data.shape, labels.shape)
print(data)
print(labels)
#%%
data = data.reshape(len(data), -1)
#%% Data cleaning
data, labels = (dc.Pruner(strategy=['lof', 'iforest'],
                          device='cpu',
                          engine='sklearn',
                          verbose=2,
                          iforest={'contamination': 'auto', 'bootstrap': True, 'n_jobs': -1, 'random_state': 1},
                          lof={'n_neighbors': 20, 'contamination': 'auto', 'n_jobs': -1}
                          )
                .fit_transform(data, labels))
#%%
data = data.values.reshape(-1, pixels_per_side, pixels_per_side, 3)
labels = labels.values
cat_labels = F.one_hot(torch.tensor(labels, requires_grad=False), num_classes=10).numpy()
#%%
# TODO: AFTER DIVISION BETWEEN TRAIN AND VALIDATION. TRY DIFFERENT STRATS TO DETERMINE THE BEST ONE
strats = ['ADASYN','BorderlineSMOTE','SVMSMOTE','KMeansSmote']
data, labels = (dc.Balancer(strategy=strats[0],
                            n_jobs=-1,verbose=2,random_state=1,
                            )
                .fit_transform(data, labels))
