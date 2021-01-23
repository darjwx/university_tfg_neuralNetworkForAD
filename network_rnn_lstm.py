# Utils and dataloader
from utils.dataloader_seq import DataLoaderSeq
from utils.transforms import Rescale, ToTensor, Normalize

# Metrics
from utils.metrics import get_metrics, show_predicted_data, visualize_model

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
input_size = 84
num_layers = 2
hidden_size = 128
num_epochs = 5
batch_size = 1
learning_rate = 0.001
num_classes = 3

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
composed = transforms.Compose([Rescale((225,400), True),
                              ToTensor(True)])

class CNNtoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNtoLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 97, 120)
        self.fc2 = nn.Linear(120, 84)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden = (torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device, requires_grad=True),
                  torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device, requires_grad=True))

        x = x.reshape(-1, 3, 225, 400)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 97)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(batch_size, sequence_length, -1)
        out1, _ = self.lstm(x, hidden)
        out1 = out1.reshape(-1, hidden_size)
        out1 = self.fc3(out1)
        out2, _ = self.lstm(x, hidden)
        out2 = out2.reshape(-1, hidden_size)
        out2 = self.fc4(out2)

        return out1, out2


# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNtoLSTM(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Custom Dataloader for NuScenes
HOME_ROUTE = '/media/darjwx/ssd_data/data/sets/nuscenes/'
dataset_train = DataLoaderSeq(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderSeq(HOME_ROUTE, 'val', 1111, 850, composed)
seq_length_train = dataset_train.get_seq_length()
seq_length_val = dataset_val.get_seq_length()

classes_speed = ['stop', 'stoping', 'accel']

classes_steering = ['straight', 'left', 'right']

trainloader = DataLoader(dataset_train, batch_size, shuffle=False)
valloader = DataLoader(dataset_val, batch_size, shuffle=False)

print('Training with %d groups of connected images' % (len(dataset_train)))
running_loss = 0.0

for epoch in range(num_epochs):

    for i, data in enumerate(trainloader):

        optimizer.zero_grad()
        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

        sequence_length = seq_length_train[i]
        out1, out2 = model(images)

        visualize_model(model, images)

        labels = labels.reshape(-1, 2)
        loss1 = criterion(out1, labels[:, 0])
        loss2 = criterion(out2, labels[:, 1])
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        #print(x.shape)

        running_loss += loss.item()

        # print every 100 groups
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f'
                 % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


print('Finished training')

print('Validating with %d groups of connected images' % (len(dataset_val)))

class_correct_1 = list(0. for i in range(3))
class_total_1 = list(0. for i in range(3))
class_correct_2 = list(0. for i in range(3))
class_total_2 = list(0. for i in range(3))

correct_total_1 = 0
correct_total_2 = 0

all_preds_1 = torch.tensor([])
all_preds_2 = torch.tensor([])
all_labels_1 = torch.tensor([])
all_labels_2 = torch.tensor([])
all_preds_1 = all_preds_1.to(device)
all_preds_2 = all_preds_2.to(device)
all_labels_1 = all_labels_1.to(device)
all_labels_2 = all_labels_2.to(device)

with torch.no_grad():
    for v, data in enumerate(valloader):

        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

        sequence_length = seq_length_val[v]
        labels = labels.reshape(-1, 2)

        out1, out2 = model(images)
        _, predicted_1 = torch.max(out1.data, 1)
        correct_1 = (predicted_1 == labels[:, 0]).squeeze()
        _, predicted_2 = torch.max(out2.data, 1)
        correct_2 = (predicted_2 == labels[:, 1]).squeeze()

        correct_total_1 += (predicted_1 == labels[:, 0]).sum().item()
        correct_total_2 += (predicted_2 == labels[:, 1]).sum().item()

        all_preds_1 = torch.cat((all_preds_1, predicted_1), dim=0)
        all_preds_2 = torch.cat((all_preds_2, predicted_2), dim=0)

        all_labels_1 = torch.cat((all_labels_1, labels[:, 0]), dim=0)
        all_labels_2 = torch.cat((all_labels_2, labels[:, 1]), dim=0)

        for i in range(np.shape(labels)[0]):
            label = labels[i,0]
            class_correct_1[label] += correct_1[i].item()
            class_total_1[label] += 1

            label = labels[i,1]
            class_correct_2[label] += correct_2[i].item()
            class_total_2[label] += 1

accuracy_t_1 = correct_total_1 / len(valloader.dataset)
accuracy_t_2 = correct_total_2 / len(valloader.dataset)
print('Accuracy 1: %1.3f  Accuracy 2: %1.3f'
      % (accuracy_t_1, accuracy_t_2))

#Per class statistics
#Accuracy
for i in range(3):
    accuracy_speed = class_correct_1[i] / class_total_1[i]
    accuracy_steering = class_correct_2[i] / class_total_2[i]
    print('Accuracy of %5s: %1.3f'
          % (classes_speed[i], accuracy_speed))
    print('Accuracy of %5s: %1.3f'
          % (classes_steering[i], accuracy_steering))

#Recall, precision, f1_score and confusion_matrix
get_metrics(all_labels_1.cpu(), all_preds_1.cpu(), 3, classes_speed)
get_metrics(all_labels_2.cpu(), all_preds_2.cpu(), 3, classes_steering)