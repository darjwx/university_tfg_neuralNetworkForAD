# Utils and dataloader
from utils.dataloader_seq import DataLoaderSeq
from utils.transforms import Rescale, ToTensor, Normalize

# Metrics
from utils.metrics import get_metrics, show_predicted_data, update_scalar_tb, pr_curve_tb, dummy_classifier

# Pytorch
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# OpenCV
import cv2 as cv

# numpy
import numpy as np

# Random generator seed
torch.manual_seed(1)

# Parameters
input_size = 84
num_layers = 1
hidden_size = 128
num_epochs = 15
batch_size = 1
learning_rate = 0.001
num_classes = 3

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (97.7419, 99.9757, 98.8718)
std = (56.8975, 55.1809, 55.8246)
composed = transforms.Compose([Rescale((225,400), True),
                              ToTensor(True),
                              Normalize(mean, std, True)])

class CNNtoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNtoLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Conv layers
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 53 * 97, 120)
        self.fc2 = nn.Linear(120, 84)

        # LSTM and output linear layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        x = x.view(-1, 3, 225, 400)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 53 * 97)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(batch_size, x.size(0), -1)
        x, _ = self.lstm(x, (h0, c0))
        out1 = self.fc3(x.view(-1, hidden_size))
        out2 = self.fc4(x.view(-1, hidden_size))

        return out1, out2


# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNtoLSTM(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

weights = torch.tensor([1., 6.67, 6.29], device=device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 2, verbose = True)

# Custom Dataloader for NuScenes
HOME_ROUTE = '/media/darjwx/ssd_data/data/sets/nuscenes/'
dataset_train = DataLoaderSeq(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderSeq(HOME_ROUTE, 'val', 1111, 850, composed)

classes_speed = ['maintain', 'stoping', 'accel']

classes_steering = ['straight', 'left', 'right']

trainloader = DataLoader(dataset_train, batch_size, shuffle=True)
valloader = DataLoader(dataset_val, batch_size, shuffle=False)

print('Training with %d groups of connected images' % (len(dataset_train)))

for epoch in range(num_epochs):
    rloss1 = 0.0
    rloss2 = 0.0

    for i, data in enumerate(trainloader):

        model.zero_grad()
        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

        out1, out2 = model(images)

        labels = labels.view(-1, 2)
        loss1 = criterion(out1, labels[:, 0])
        loss2 = criterion(out2, labels[:, 1])
        loss = loss1 + loss2

        update_scalar_tb('training loss speed', loss1, epoch * len(trainloader) + i)
        update_scalar_tb('training loss direction', loss2, epoch * len(trainloader) + i)

        loss.backward()
        optimizer.step()

        rloss1 += loss1.item()
        rloss2 += loss2.item()

        # print every 100 groups
        if i % 100 == 99:
            print('[%d, %5d] loss speed: %.3f loss direction: %.3f'
                 % (epoch + 1, i + 1, rloss1 / 100, rloss2 / 100))

            rloss1 = 0.0
            rloss2 = 0.0

    # Validation loss
    with torch.no_grad():
        for i, data in enumerate(valloader):
            images = data['image']
            labels = data['label']

            images = images.to(device)
            labels = labels.to(device)

            labels = labels.view(-1, 2)

            out1, out2 = model(images)
            loss1_val = criterion(out1, labels[:, 0])
            loss2_val = criterion(out2, labels[:, 1])

            update_scalar_tb('validation loss speed', loss1_val, epoch * len(valloader) + i)
            update_scalar_tb('validation loss direction', loss2_val, epoch * len(valloader) + i)


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

preds_1 = []
preds_2 = []

with torch.no_grad():
    for v, data in enumerate(valloader):

        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

        labels = labels.view(-1, 2)

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

        class_1_predictions = [F.softmax(output, dim=0) for output in out1]
        class_2_predictions = [F.softmax(output, dim=0) for output in out2]
        preds_1.append(class_1_predictions)
        preds_2.append(class_2_predictions)

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

# Per class statistics
# Dummy classifier: most_frequent and constant
print('Dummy classifier 1')
dummy_classifier(all_labels_1.cpu(), 1)
print('Dummy classifier 2')
dummy_classifier(all_labels_2.cpu(), 1)

# Accuracy
for i in range(3):
    accuracy_speed = class_correct_1[i] / class_total_1[i]
    accuracy_steering = class_correct_2[i] / class_total_2[i]
    print('Accuracy of %5s: %1.3f'
          % (classes_speed[i], accuracy_speed))
    print('Accuracy of %5s: %1.3f'
          % (classes_steering[i], accuracy_steering))

# p-r curve
preds_1 = torch.cat([torch.stack(batch) for batch in preds_1])
preds_2 = torch.cat([torch.stack(batch) for batch in preds_2])

pr_curve_tb(3, all_labels_1, all_labels_2, preds_1, preds_2)

# Recall, precision, f1_score and confusion_matrix
get_metrics(all_labels_1.cpu(), all_preds_1.cpu(), 3, classes_speed)
get_metrics(all_labels_2.cpu(), all_preds_2.cpu(), 3, classes_steering)
