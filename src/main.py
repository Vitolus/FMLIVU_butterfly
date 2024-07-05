#%%
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
#%%
# building the dataset
data = []
labels = []
path = '../data/images'
folder = os.listdir(path)
for file in folder:
    label = int(file[:3])
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    data.append(img)
    labels.append(label)

dict_1 = {1: 'Danaus_plexippus',
          2: 'Heliconius_charitonius',
          3: 'Heliconius_erato',
          4: 'Junonia_coenia',
          5: 'Lycaena_phlaeas',
          6: 'Nymphalis_antiopa',
          7: 'Papilio_cresphontes',
          8: 'Pieris_rapae',
          9: 'Vanessa_atalanta',
          10: 'Vanessa_cardui'}
list_labels = []
for i in labels:
    list_labels.append(dict_1[i])

#display amount of images per class
df = pd.DataFrame(list_labels, columns=['labels'])
df['labels'].value_counts().plot(kind='bar', colormap='viridis')
plt.show()
