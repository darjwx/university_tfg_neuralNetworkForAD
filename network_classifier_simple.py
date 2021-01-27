# Utils and dataloader
from utils.dataloader_hf import DataLoaderHF
from utils.transforms import Rescale, ToTensor, Normalize
from utils.metrics import get_metrics, show_predicted_data, update_scalar_tb, pr_curve_tb

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
# Original resolution / 4 (900, 1600) (h, w)
mean = (97.7419, 99.9757, 98.8718)
std = (56.8975, 55.1809, 55.8246)
composed = transforms.Compose([Rescale((225,400)),
                              ToTensor(),
                              Normalize(mean, std)])
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 97, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.fc4 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 97)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out1 = self.fc3(x)
        out2 = self.fc4(x)
        return out1, out2

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Custom Dataloader for NuScenes
HOME_ROUTE = '/media/darjwx/ssd_data/data/sets/nuscenes/'
dataset_train = DataLoaderHF(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderHF(HOME_ROUTE, 'val', 1111, 850, composed)

classes_speed = ['stop', 'stoping', 'accel']

classes_steering = ['straight', 'left', 'right']

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

        out1, out2 = model(images)

        loss1 = criterion(out1, labels[:, 0])
        loss2 = criterion(out2, labels[:, 1])
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f'
                 % (epoch + 1, i + 1, running_loss / 1000))

            update_scalar_tb('training loss', running_loss / 1000, epoch * len(trainloader) + i)
            running_loss = 0.0

print('Finished training')

# Uncomment to save the model
# PATH = './models/class_net.pth'
# torch.save(model.state_dict(), PATH)

print('Validating with %d' % (len(dataset_val)))

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
    for data in valloader:
        images = data['image']
        labels = data['label']

        images = images.to(device)
        labels = labels.to(device)

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

#Per class statistics
#Accuracy
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

#Recall, precision, f1_score and confusion_matrix
get_metrics(all_labels_1.cpu(), all_preds_1.cpu(), 3, classes_speed)
get_metrics(all_labels_2.cpu(), all_preds_2.cpu(), 3, classes_steering)

#Show examples
show_predicted_data(valloader, classes_speed, classes_steering, all_preds_1.cpu(), all_preds_2.cpu())
