#%% import libraries
import os
import cv2
import random
from PIL import Image
from collections import Counter
import optuna
from optuna.trial import TrialState
from tqdm.notebook import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
path = 'data/images'
pixels_per_side = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)
#%%
data = []
labels = []
folder = os.listdir(path)
for file in folder:
    img = cv2.imread(str(os.path.join(path, file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (pixels_per_side, pixels_per_side))
    data.append(img)
    labels.append(int(file[:3]) - 1)
data = np.array(data)
labels = np.array(labels)
classes = ['Danaus plexippus', 'Heliconius charitonius', 'Heliconius erato', 'Junonia coenia', 'Lycaena phlaeas',
           'Nymphalis antiopa', 'Papilio cresphontes', 'Pieris rapae', 'Vanessa atalanta', 'Vanessa cardui']
print(data.shape, labels.shape)
print([classes[labels[i]] for i in range(10)])
#%%
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=np.random.randint(0,len(labels))
        ax[i,j].imshow(data[l])
        ax[i,j].set_title(str(classes[labels[l]]))
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
df = pd.DataFrame(labels, columns=['labels'])
df['labels'] = [classes[i] for i in df['labels']]
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = [cycle[i % len(cycle)] for i in range(len(df['labels'].unique()))]
df['labels'].value_counts().plot(kind='bar', color=colors)
plt.xlabel('Species')
plt.ylabel('Counts')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
data_min = np.min(data, axis=(0, 1, 2), keepdims=True) # data_min contains the minimum value of each channel
data_max = np.max(data, axis=(0, 1, 2), keepdims=True) # data_max contains the maximum value of each channel
data_scaled = (data - data_min) / (data_max - data_min) # data_scaled contains the scaled data between 0 and 1
mean = np.mean(data_scaled, axis=(0, 1, 2)) # mean contains the mean of each channel
std = np.std(data_scaled, axis=(0, 1, 2)) # std contains the standard deviation of each channel
print(mean)
print(std)
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
        if self.transform:
            data = self.transform(data)
        labels = torch.tensor(self.labels[idx])
        return data, labels
#%% Data augmentation
class_counts = Counter(labels)
target_count = 100
samples_needed = {cls: target_count - count for cls, count in class_counts.items()}
aug_data, aug_labels = [], []

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])
#%%
print(len(data), len(labels))
print(type(data), type(labels))
print(type(data[0]), type(labels[0]))
print(data[0].shape)
#%%
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
while any(samples_needed[cls] > 0 for cls in samples_needed):
    idx = random.randint(0, len(X_train) - 1)
    img, label = X_train[idx], y_train[idx]
    if samples_needed[label] > 0:
        img = Image.fromarray(img)
        img = transform(img)
        aug_data.append(img)
        aug_labels.append(label)
        samples_needed[label] -= 1
X_train = np.concatenate([X_train, aug_data], axis=0)
y_train = np.concatenate([y_train, aug_labels], axis=0)
#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trainset = MyDataset(X_train, y_train, transform=transform)
testset = MyDataset(X_test, y_test, transform=transform)
#%%
class EarlyStopping:
    def __init__(self, patience=10, delta=0.05, window_size=5):
        self.patience = patience
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.delta = delta
        self.window_size = window_size
        self.val_window = []

    def __call__(self, val_loss, net):
        self.val_window.append(val_loss)
        if len(self.val_window) > self.window_size:
            self.val_window.pop(0)
        avg_val = np.mean(self.val_window)

        if avg_val == self.best_score or avg_val > self.best_score + self.delta:
            self.counter += 1
        elif avg_val < self.best_score:
            self.best_score = avg_val
            self.save_checkpoint(avg_val, net)
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val, model):
        torch.save(model.state_dict(), '../models/checkpoint.pth')
#%%
def fit(net, trainloader, optimizer, loss_fn=nn.CrossEntropyLoss()):
    net.train()
    total_loss, acc, count = 0, 0, 0
    for features, labels in trainloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = net(features)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum().item()
        count += len(labels)
    return total_loss / count, acc / count

def predict(net, valloader, loss_fn=nn.CrossEntropyLoss()):
    net.eval()
    count, acc, total_loss = 0, 0, 0
    with torch.no_grad():
        for features, labels in valloader:
            features = features.to(device)
            labels = labels.to(device)
            count += len(labels)
            out = net(features)
            total_loss += loss_fn(out, labels).item()
            pred = torch.max(out, 1)[1]
            acc += (pred == labels).sum().item()
    return total_loss / count, acc / count

def objective(trial, trainset, X, y):
    lr = trial.suggest_float('lr', 0.0009, 0.002, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    val_accs = []
    train_acc, train_loss, val_acc, val_loss, mean_acc = 0, 0, 0, 0, 0
    split_num = 0
    prog_bar = tqdm(skf.split(X, y), desc="Splits")
    for train_index, val_index in prog_bar:
        split_num += 1
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index))
        valloader = DataLoader(trainset, batch_size=256, sampler=SubsetRandomSampler(val_index))
        net = Net().to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        early_stopping = EarlyStopping(patience=10, delta=0.005, window_size=5)

        for epoch in range(100):
            train_loss, train_acc = fit(net, trainloader, optimizer)
            val_loss, val_acc = predict(net, valloader)
            scheduler.step(val_acc)
            early_stopping(val_acc, net)
            prog_bar.set_description(
                f"Split {split_num} - Epoch {epoch + 1}, Train acc={train_acc:.3f}, Train loss={train_loss:.3f}, "
                f"Validation acc={val_acc:.3f}, Validation loss={val_loss:.3f}")
            if early_stopping.early_stop:
                break

        val_accs.append(val_acc)
        mean_acc = np.mean(val_accs)
        trial.report(mean_acc, split_num)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return mean_acc
#%% CNN definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 11, 1)
        self.conv3 = nn.Conv2d(32, 64, 5, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flat(x)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = self.fc2(x)
        return x
#%%
X = np.zeros(len(trainset))
labelloader =  DataLoader(trainset, batch_size=256, shuffle=False)
y = []
for _, label in labelloader:
    y.append(label.numpy())
y = np.concatenate(y, axis=0)
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(lambda trial: objective(trial, trainset, X, y), n_trials=10)
#%%
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
#%%
writer = SummaryWriter("runs")
net = Net().to(device)
writer.add_graph(net, torch.zeros((1, 3, pixels_per_side, pixels_per_side)).to(device))
writer.flush()
summary(net, input_size=(1, 3, pixels_per_side, pixels_per_side))
#%%
trainloader = DataLoader(trainset, batch_size=trial.params['batch_size'], shuffle=True)
testloader = DataLoader(testset, batch_size=256, shuffle=False)
optimizer = optim.Adam(net.parameters(), lr=trial.params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
early_stopping = EarlyStopping(patience=10, delta=0.005, window_size=5)
train_accs, train_losses, test_accs, test_losses = [], [], [], []
prog_bar = tqdm(range(100), total=100)
for epoch in prog_bar:
    train_loss, train_acc = fit(net,trainloader, optimizer)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_loss, test_acc = predict(net, testloader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    scheduler.step(test_acc)
    early_stopping(test_acc, net)
    prog_bar.set_description(f"Epoch {epoch + 1}, Train acc={train_acc:.3f}, Train loss={train_loss:.3f}, "
                             f"Test acc={test_acc:.3f}, Test loss={test_loss:.3f}")
    if early_stopping.early_stop:
        print("Early stopping")
        break
#%% evaluate the model
train_accs = np.array(train_accs)
train_losses = np.array(train_losses)
test_accs = np.array(test_accs)
test_losses = np.array(test_losses)
print(train_losses.shape, test_losses.shape)
print(train_losses, test_losses)
print(train_accs.shape, test_accs.shape)
print(train_accs, test_accs)
#%%
plt.figure()
plt.plot(train_accs, label='Train acc')
plt.plot(test_accs, label='Test acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.figure()
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(bottom=0)
plt.legend()
plt.show()
