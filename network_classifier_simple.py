# Dataloader
from utils.dataloader_hf import DataLoaderHF
from utils.transforms import Rescale, ToTensor

# Pytorch
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# OpenCV
import cv2 as cv

# numpy
import numpy as np

# Parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Transforms
# Original resolution / 4 (1600, 900)
composed = transforms.Compose([Rescale((400,225)),
                              ToTensor()])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 97, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 97)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Custom Dataloader for NuScenes
HOME_ROUTE = ''
dataset_train = DataLoaderHF(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderHF(HOME_ROUTE, 'val', 1111, 850, composed)

trainloader = DataLoader(dataset_train, batch_size, shuffle=False, num_workers=4)
valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=4)

print('Training with %d images' % (len(dataset_train)))
running_loss = 0.0
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader):
        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished training')

# Uncomment to save the model
# PATH = './models/class_net.pth'
# torch.save(model.state_dict(), PATH)

print('Validating with %d' % (len(dataset_val)))
correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network: %d %%'
       % (100 * correct / total))
