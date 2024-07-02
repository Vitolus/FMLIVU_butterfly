#%%
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np

#%%
# hyperparameters
batch_size = 128
lr = 0.01
n_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%
data_train = torchvision.datasets.MNIST('./data', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
data_test = torchvision.datasets.MNIST('./data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)


#%%
def plot_results(hist):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['val_acc'], label='Validation acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['val_loss'], label='Validation loss')
    plt.legend()


def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    total_loss, acc, count = 0, 0, 0
    for features, labels in dataloader:
        optimizer.zero_grad()
        count += len(labels)
        labels = labels.to(device)
        out = net(features.to(device))
        loss = loss_fn(out, labels)  #cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)
        acc += (predicted == labels).sum()
    return total_loss.item() / count, acc.item() / count


def validate(net, dataloader, loss_fn=nn.NLLLoss()):
    net.eval()
    count, acc, loss = 0, 0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            count += len(labels)
            labels = labels.to(device)
            out = net(features.to(device))
            loss += loss_fn(out, labels)
            pred = torch.max(out, 1)[1]
            acc += (pred == labels).sum()
    return loss.item() / count, acc.item() / count


def train(net, train_loader, test_loader, optimizer=None, lr=0.01, epochs=10, loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    res = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
        tl, ta = train_epoch(net, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn)
        vl, va = validate(net, test_loader, loss_fn=loss_fn)
        print(f"Epoch {ep+1:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res


#%%
class OneConv(nn.Module):
    def __init__(self):
        super(OneConv, self).__init__()
        self.conv = nn.Conv2d(1, 9, kernel_size=(5, 5))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(9 * 24 * 24, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = self.flatten(x)
        x = nn.functional.log_softmax(self.fc(x), dim=1)
        return x


net = OneConv().to(device)
summary(net, (1, 1, 28, 28))


#%%
hist = train(net, train_loader, test_loader, epochs=n_epochs, lr=lr)
plot_results(hist)
#%%
# plot the 9 filters
fig, ax = plt.subplots(1, 9)
with torch.no_grad():
    p = next(net.conv.parameters())
    for i,x in enumerate(p):
        ax[i].imshow(x.detach().cpu()[0, ...])
        ax[i].axis('off')
