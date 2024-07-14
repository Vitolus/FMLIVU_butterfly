#%% import libraries
import os
import cv2
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)
#%% building the dataset
data = []
labels = []
path = 'data/images'
folder = os.listdir(path)
for file in folder:
    img = cv2.imread(str(os.path.join(path, file)))
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
data = data.reshape(len(data), -1)
labels = np.array(labels)
labels = np.array(labels).reshape(-1, 1)
# categorical_labels = F.one_hot(torch.tensor(labels, requires_grad=False), num_classes=10).numpy()
#%% revert flatten data
data = data.reshape(len(data), 224, 224, 3)
#%%
print(data.shape, labels.shape)
print(data)
print(labels)
#%%
clf = atom.ATOMClassifier(data, y=labels,
                          test_size=0.2,
                          device='cpu',
                          engine='sklearn',
                          n_jobs=-1, verbose=2, random_state=1)
#%% Data cleaning
data, labels = (dc.Pruner(strategy=['lof', 'iforest'],
                          device='cpu',
                          engine='sklearn',
                          verbose=2,
                          iforest={'n_estimators': 100,
                                   'contamination': 'auto',
                                   'bootstrap': True,
                                   'n_jobs': -1, 'verbose': 2, 'random_state': 1},
                          lof={'n_neighbors': 20,
                               'metric': 'minkowski',
                               'contamination': 'auto',
                               'n_jobs': -1}
                          )
                .fit_transform(data, labels))
#%%
# TODO: AFTER DIVISION BETWEEN TRAIN AND VALIDATION
strats = ['ADASYN','BorderlineSMOTE','SVMSMOTE','KMeansSmote']
data, labels = (dc.Balancer(strategy=strats[0],
                            n_jobs=-1,verbose=2,random_state=1,
                            )
                .fit_transform(data, labels))
