#%% import libraries
import os
import cv2
from PIL import Image
import optuna
import atom
import atom.data_cleaning as dc
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchinfo import summary
from sklearn.model_selection import StratifiedKFold
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
classes = ['Danaus plexippus', 'Heliconius charitonius', 'Heliconius erato', 'Junonia coenia', 'Lycaena phlaeas',
           'Nymphalis antiopa', 'Papilio cresphontes', 'Pieris rapae', 'Vanessa atalanta', 'Vanessa cardui']
print([classes[labels[i]] for i in range(10)])
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
        labels = torch.tensor(self.labels[idx])
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
#%%
dataset = MyDataset(data, labels)
trainset, testset = random_split(dataset, [0.85, 0.15])
testloader = DataLoader(testset, batch_size=64, shuffle=False)
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(120 * 49 * 49, 64) # ((224 -4)/2 -4)/2 -4
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.flat(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)
summary(net, input_size=(1, 3, pixels_per_side, pixels_per_side))
#%%
def fit(net, trainloader, optimizer, loss_fn=nn.CrossEntropyLoss()):
    net.train()
    total_loss = acc = count = 0
    # TODO: error on index in train loader
    for features, labels in trainloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(logits, 1)
        acc += (predicted == labels).sum()
        count += len(labels)
    return total_loss.item() / count, acc.item() / count

def predict(net, valloader, loss_fn):
    net.eval()
    count = acc = total_loss = 0
    with torch.no_grad():
        for features, labels in valloader:
            features = features.to(device)
            labels = labels.to(device)
            count += len(labels)
            logits = net(features)
            total_loss += loss_fn(logits, labels)
            pred = torch.max(logits, 1)[1]
            acc += (pred == labels).sum()
    return total_loss.item() / count, acc.item() / count

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epochs = trial.suggest_int('epochs', 5, 50)
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    val_accs = []
    for train_index, val_index in skf.split(trainset.dataset.data, trainset.dataset.labels):
        train_data = Subset(trainset, train_index)
        val_data = Subset(trainset, val_index)
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            train_loss, train_acc = fit(net, trainloader, optimizer, criterion)
            val_loss, val_acc = predict(net, valloader, criterion)
            print(f"Epoch {epoch + 1:2}, Train acc={train_acc:.3f}, Val acc={val_acc:.3f}, Train loss={train_loss:.3f}, Val loss={val_loss:.3f}")
        val_accs.append(val_acc)
    return np.mean(val_accs)

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial), n_trials=100)
#%%
strategy = trial.suggest_categorical('strategy', ['ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSmote'])
X_train = trainset.dataset.data[train_index]
        X_train = X_train.reshape(len(X_train), -1)
        y_train = trainset.dataset.labels[train_index]
        X_train, y_train = (dc.Balancer(strategy=strategy,
                                        n_jobs=-1,verbose=2,
                                        random_state=1,)
                            .fit_transform(X_train, y_train))
        train_data = MyDataset(X_train, y_train)