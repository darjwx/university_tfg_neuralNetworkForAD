"""
    Simple classifier - Classifier designed to be light.
    Copyright (C) 2020-2021  Darío Jiménez

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <http://www.gnu.org/licenses/>.
"""

# Utils and dataloader
from utils.dataloader_hf import DataLoaderHF
from utils.transforms import Rescale, ToTensor, Normalize
from utils.metrics import get_metrics, show_predicted_data, update_scalar_tb, pr_curve_tb, dummy_classifier

# Pytorch
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# OpenCV
import cv2 as cv

# numpy
import numpy as np

# Argument parser
import argparse

# Random generator seed
torch.manual_seed(1)

# Configurations
def str_to_bool(arg):
    if arg.lower() in ['y', 'true', '1']:
        return True
    elif arg.lower() in ['n', 'false', '0']:
        return False
    else:
        print('Wrong value')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, default=10, help='Batch size')
parser.add_argument('--res', nargs=2, type=int, default=[225,400], help='Images resolution')
parser.add_argument('--weights', nargs=3, type=float, default=[1., 1., 1.], help='Loss weights')
parser.add_argument('--canbus', type=str_to_bool, default=False, help='Wheter to use canbus data as an input')
parser.add_argument('--route', type=str, default='/data/sets/nuscenes/', help='Route where the NuScenes dataset is located')
parser.add_argument('--tb', type=str, default='None', help='Path for the TensorBoard logs')
parser.add_argument('--save', type=str, default='None', help='Location where the model is going to be saved')
parser.add_argument('--load', type=str, default='None', help='Path to the model to be loaded')

args = parser.parse_args()

# Parameters
num_epochs = args.epochs
batch_size = args.batch
learning_rate = args.lr
if args.canbus:
    add_dim = 2
else:
    add_dim = 0

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (0.3833, 0.3921, 0.3877)
std = (0.2231, 0.2164, 0.2189)
composed = transforms.Compose([Rescale(tuple(args.res), canbus=args.canbus),
                              ToTensor(canbus=args.canbus),
                              Normalize(mean, std, canbus=args.canbus)])
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 97, 120)
        self.fc2 = nn.Linear(120 + add_dim, 84)
        self.fc3 = nn.Linear(84, 3)
        self.fc4 = nn.Linear(84, 3)

    def forward(self, x, d=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 97)
        x = F.relu(self.fc1(x))

        if args.canbus:
            x = torch.cat((x, d), dim=1)

        x = F.relu(self.fc2(x))
        out1 = self.fc3(x)
        out2 = self.fc4(x)
        return out1, out2

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model = model.to(device)

weights = torch.from_numpy(np.array(args.weights)).float().to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'max', 0.1, 2, verbose = True)

# Custom Dataloader for NuScenes
HOME_ROUTE = args.route
dataset_train = DataLoaderHF(HOME_ROUTE, 'train', 1111, 850, composed, args.canbus)
dataset_val = DataLoaderHF(HOME_ROUTE, 'val', 1111, 850, composed, args.canbus)

classes_speed = ['maintain', 'stoping', 'accel']

classes_steering = ['straight', 'left', 'right']

trainloader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=4)

if args.load != 'None':
    print('Loading model from %s' % (args.load))
    model.load_state_dict(torch.load(args.load))
else:
    print('Training with %d images' % (len(dataset_train)))
    for epoch in range(num_epochs):
        rloss1 = 0.0
        rloss2 = 0.0
        for i, data in enumerate(trainloader):
            images = data['image']
            labels = data['label']
            images = images.to(device)
            labels = labels.to(device)

            if args.canbus:
                ndata = data['numerical']
                ndata = ndata.to(device)
                out1, out2 = model(images, ndata)
            else:
                out1, out2 = model(images)

            loss1 = criterion(out1, labels[:, 0])
            loss2 = criterion(out2, labels[:, 1])
            loss = loss1 + loss2

            if args.tb != 'None':
                update_scalar_tb('training loss speed', loss1, epoch * len(trainloader) + i, args.tb)
                update_scalar_tb('training loss direction', loss2, epoch * len(trainloader) + i, args.tb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rloss1 += loss1.item()
            rloss2 += loss2.item()

            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss speed: %.3f  loss direction: %.3f'
                     % (epoch + 1, i + 1, rloss1 / 1000, rloss2 / 1000))

                rloss1 = 0.0
                rloss2 = 0.0

        # Val acc for the scheduler step
        with torch.no_grad():
            correct1 = 0
            correct2 = 0
            for data in valloader:
                images = data['image']
                labels = data['label']
                images = images.to(device)
                labels = labels.to(device)

                if args.canbus:
                    ndata = data['numerical']
                    ndata = ndata.to(device)
                    out1, out2 = model(images, ndata)
                else:
                    out1, out2 = model(images)

                _, predicted_1 = torch.max(out1.data, 1)
                _, predicted_2 = torch.max(out2.data, 1)
                correct1 += (predicted_1 == labels[:, 0]).sum().item()
                correct2 += (predicted_2 == labels[:, 1]).sum().item()

            acc1 = correct1 / len(valloader)
            acc2 = correct2 / len(valloader)
            acc = (acc1 + acc2) / 2

            scheduler.step(acc)


    print('Finished training')

# Save model
if args.save != 'None':
    torch.save(model.state_dict(), args.save)

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

        if args.canbus:
            ndata = data['numerical']
            ndata = ndata.to(device)
            out1, out2 = model(images, ndata)
        else:
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
# Dummy classifier: most_frequent and constant
print('Dummy classifier 1')
dummy_classifier(all_labels_1.cpu(), 1)
print('Dummy classifier 2')
dummy_classifier(all_labels_2.cpu(), 1)

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

if args.tb != 'None':
    pr_curve_tb(3, all_labels_1, all_labels_2, preds_1, preds_2, args.tb)

#Recall, precision, f1_score and confusion_matrix
get_metrics(all_labels_1.cpu(), all_preds_1.cpu(), 3, classes_speed)
get_metrics(all_labels_2.cpu(), all_preds_2.cpu(), 3, classes_steering)

#Show examples
show_predicted_data(valloader, classes_speed, classes_steering, all_preds_1.cpu(), all_preds_2.cpu(), mean, std)
