#%% import libraries
import os
import cv2
from atom import ATOMClassifier
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%% building the dataset
data = []
labels = []
path = 'data/images'
folder = os.listdir(path)
for file in folder:
    img = cv2.imread(os.path.join(path, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data.append(img)
    labels.append(int(file[:3]) - 1)

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
data = np.array(data) / 255.0
labels = np.array(labels)
flatten_data = data.reshape(len(data), -1)
categorical_labels = F.one_hot(torch.tensor(labels, requires_grad=False), num_classes=10).numpy()
#%% Data cleaning
atom = ATOMClassifier(flatten_data, categorical_labels,
                      test_size=0.2,
                      n_jobs=-1,
                      device=('gpu' if device == 'cuda' else 'cpu'),
                      engine='cuml',
                      verbose=2,
                      random_state=1)
