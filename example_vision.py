#%%
import torch
import torch.nn as nn
from torch.nn.functional import relu, log_softmax
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchinfo import summary

#%%
data_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=ToTensor())
data_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=ToTensor())
# if using own dataset, use OpenCV library to load images and convert them to tensors using ToTensor() (which scales the image to [0, 1])
# better to use torchvision.datasets.ImageFolder() for custom datasets (which performs all preprocessing)
#%%
# visualize dataset
fig, ax = plt.subplots(1, 7)
for i in range(7):
    ax[i].imshow(data_train[i][0].view(28, 28))
    ax[i].set_title(data_train[i][1])
    ax[i].axis('off')
#%%
# dataset stracture exploration
print('Training samples:', len(data_train))
print('Test samples:', len(data_test))
print('Tensor size:', data_train[0][0].size())
print('First 10 labels:', [data_train[i][1] for i in range(10)])
print('Min intensity:', data_train[0][0].min().item())
print('Max intensity:', data_train[0][0].max().item())
#%%
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10),
    nn.LogSoftmax(dim=0)
)
print('Digit to be predicted:', data_train[0][1])
torch.exp(net(data_train[0][0]))
#%%
train_loader = torch.utils.data.DataLoader(data_train, batch_size=64)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=64)


#%%
def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()  # set the model to training mode
    total_loss, acc, count = 0, 0, 0
    for features, labels in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = loss_fn(out, labels)  # cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        _, predicted = torch.max(out, 1)  # (value, label)
        acc += (predicted == labels).sum()
        count += len(labels)
    return total_loss.item() / count, acc.item() / count  # average loss, accuracy


train_epoch(net, train_loader)


#%%
def validate(net, dataloader, loss_fn=nn.NLLLoss()):
    net.eval()  # set the model to evaluation (test) mode
    count = acc = loss = 0
    with torch.no_grad():
        for features, labels in dataloader:
            out = net(features)
            loss += loss_fn(out, labels)
            pred = torch.max(out, 1)[1]
            acc += (pred == labels).sum()
            count += len(labels)
    return loss.item() / count, acc.item() / count


validate(net, test_loader)


#%%
def train(net, train_loader, test_loader, optimizer=None, lr=0.01, epochs=10, loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    res = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for ep in range(epochs):
        tl, ta = train_epoch(net, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn)
        vl, va = validate(net, test_loader, loss_fn=loss_fn)
        print(f"Epoch {ep + 1:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res

#%%
# Re-initialize the network to start from scratch
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),  # 784 inputs, 10 outputs
    nn.LogSoftmax(dim=0))
summary(net, input_size=(1, 28, 28), device='cpu')


#%%
# visualize the training process
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


#%%
hist = train(net, train_loader, test_loader, epochs=5)
plot_results(hist)
#%%
# visualize the weights
weight_tensor = next(net.parameters())
fig, ax = plt.subplots(1, 10, figsize=(15, 4))
for i, x in enumerate(weight_tensor):
    ax[i].imshow(x.view(28, 28).detach())
#%%
# multi-layer network
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 100),  # 784 inputs, 100 outputs
    nn.ReLU(),  # Activation Function
    nn.Linear(100, 10),  # 100 inputs, 10 outputs
    nn.LogSoftmax(dim=0))
summary(net, input_size=(1, 28, 28))


#%%
# network class representation
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(784, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = relu(x)
        x = self.out(x)
        x = log_softmax(x, dim=0)
        return x


net = MyNet()
summary(net, input_size=(1, 28, 28), device='cpu')
