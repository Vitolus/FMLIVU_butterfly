#%%
import torch, torchvision, json, requests
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np

#%%
from PIL import Image
import glob
import os
import zipfile


def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except:
        return False


def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)


check_image_dir('data/PetImages/Cat/*.jpg')
check_image_dir('data/PetImages/Dog/*.jpg')

#%%
std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    std_normalize])
dataset = torchvision.datasets.ImageFolder('data/PetImages', transform=trans)
trainset, testset = torch.utils.data.random_split(dataset, [20000, len(dataset) - 20000])
#%%
file_path = 'models/vgg16-397923af.pth'
vgg = torchvision.models.vgg16()
vgg.load_state_dict(torch.load(file_path))
vgg.eval()
sample_image = dataset[0][0].unsqueeze(0)
res = vgg(sample_image)
print(res[0].argmax())
class_map = json.loads(requests.get("https://raw.githubusercontent.com/MicrosoftDocs/pytorchfundamentals/main/"
                                    "computer-vision-pytorch/imagenet_class_index.json").text)
class_map = {int(k): v for k, v in class_map.items()}
class_map[res[0].argmax().item()]


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
        print(f"Epoch {ep + 1:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res


def train_long(net, train_loader, test_loader, epochs=5, lr=0.01, optimizer=None, loss_fn=nn.NLLLoss(), print_freq=10):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        net.train()
        total_loss, acc, count = 0, 0, 0
        for i, (features, labels) in enumerate(train_loader):
            lbls = labels.to(device)
            optimizer.zero_grad()
            out = net(features.to(device))
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss
            _, predicted = torch.max(out, 1)
            acc += (predicted == lbls).sum()
            count += len(labels)
            if i % print_freq == 0:
                print("Epoch {}, minibatch {}: train acc = {}, train loss = {}"
                      .format(epoch, i, acc.item() / count, total_loss.item() / count))
        vl, va = validate(net, test_loader, loss_fn)
        print("Epoch {} done, validation acc = {}, validation loss = {}".format(epoch, va, vl))


#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Doing computations on ', device)
vgg.to(device)
sample_image = sample_image.to(device)
summary(vgg, input_size=(1, 3, 224, 224))
#%%
res = vgg.features(sample_image).cpu()
plt.figure(figsize=(15, 3))
plt.imshow(res.detach().view(-1, 512))
print(res.size())
#%%
bs = 8
dl = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
num = bs * 100
feature_tensor = torch.zeros(num, 512 * 7 * 7).to(device)
label_tensor = torch.zeros(num).to(device)
i = 0
for x, l in dl:
    with torch.no_grad():
        f = vgg.features(x.to(device))
        feature_tensor[i:i + bs] = f.view(bs, -1)
        label_tensor[i:i + bs] = l
        i += bs
        print('.', end='')
        if i >= num:
            break
#%%
vgg_dataset = torch.utils.data.TensorDataset(feature_tensor, label_tensor.to(torch.long))
train_ds, test_ds = torch.utils.data.random_split(vgg_dataset, [700, 100])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
net = torch.nn.Sequential(torch.nn.Linear(512 * 7 * 7, 2), torch.nn.LogSoftmax()).to(device)
history = train(net, train_loader, test_loader)
#%%
vgg.classifier = torch.nn.Linear(25088, 2).to(device)
for x in vgg.features.parameters():
    x.requires_grad = False
summary(vgg, (1, 3, 244, 244))
#%%
trainset, testset = torch.utils.data.random_split(dataset, [20000, len(dataset) - 20000])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16)
train_long(vgg, train_loader, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), epochs=1, print_freq=90)
#%%
torch.save(vgg,'data/cats_dogs.pth')
#%%
vgg = torch.load('data/cats_dogs.pth')
for x in vgg.features.parameters():
    x.requires_grad = True
train_long(vgg,train_loader,test_loader,loss_fn=torch.nn.CrossEntropyLoss(),epochs=1,print_freq=90,lr=0.0001)