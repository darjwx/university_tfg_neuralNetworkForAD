"""
    Regression - Regression model using LSTM layers.
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
from utils.dataloader_reg import DataLoaderReg
from utils.transforms import Rescale, ToTensor, Normalize

# Metrics
from utils.metrics import show_predicted_data, update_scalar_tb, draw_reg_lineplot, get_accuracy, mean_squared_error

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

# Argument parser
import argparse

# Random generator seed
torch.manual_seed(1)

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
parser.add_argument('--hidden', type=int, default=128, help='LSTM hidden size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--res', nargs=2, type=int, default=[225,400], help='Images resolution')
parser.add_argument('--route', type=str, default='/data/sets/nuscenes/', help='Route where the NuScenes dataset is located')
parser.add_argument('--coef', type=float, default=0.15, help='Coef for the accuracy')
parser.add_argument('--tb', type=str, default='None', help='Path for the TensorBoard logs')
parser.add_argument('--save', type=str, default='None', help='Location where the model is going to be saved')
parser.add_argument('--load', type=str, default='None', help='Path to the model to be loaded')

args = parser.parse_args()

# Parameters
input_size = 84
num_layers = args.layers
hidden_size = args.hidden
num_epochs = args.epochs
batch_size = 1
learning_rate = args.lr
output = 1
coef = args.coef

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (0.3833, 0.3921, 0.3877)
std = (0.2231, 0.2164, 0.2189)

mean_sp = 19.2159
std_sp = 3.0407
mean_st = 25.0026
std_st = 63.8881
#mean_st, std_st, mean_sp, std_sp
composed = transforms.Compose([Rescale(tuple(args.res), reg=True),
                              ToTensor(reg=True),
                              Normalize(mean, std, mean_sp, std_sp, mean_st, std_st, reg=True)])

class CNNtoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output):
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
        self.fc3 = nn.Linear(hidden_size, output)
        self.fc4 = nn.Linear(hidden_size, output)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        batch_size, sl, C, H, W = x.size()
        x = x.view(batch_size * sl, C, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 53 * 97)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = x.unsqueeze(0)
        x, _ = self.lstm(x, (h0, c0))
        x = x.view(-1, hidden_size)
        out1 = self.fc3(x)
        out2 = self.fc4(x)

        return out1, out2



# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNtoLSTM(input_size, hidden_size, num_layers, output)
model = model.to(device)

criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
# scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 2, verbose = True)

# Custom Dataloader for NuScenes
HOME_ROUTE = args.route
dataset_train = DataLoaderReg(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderReg(HOME_ROUTE, 'val', 1111, 850, composed)

trainloader = DataLoader(dataset_train, batch_size, shuffle=True)
valloader = DataLoader(dataset_val, batch_size, shuffle=False)

if args.load != 'None':
    print('Loading model from %s' % (args.load))
    model.load_state_dict(torch.load(args.load))
else:
    print('Training with %d groups of connected images' % (len(dataset_train)))

    for epoch in range(num_epochs):
        rloss1 = 0.0
        rloss2 = 0.0

        model.train()
        for i, data in enumerate(trainloader):

            model.zero_grad()
            images = data['image']
            canbus = data['can_bus']

            images = images.to(device)
            canbus = canbus.to(device)
            canbus = canbus.squeeze(0)

            out1, out2 = model(images)

            # canbus[:,0] -> (42,) -- unsqueeze(1) -> (42,1)
            loss1 = criterion(out1, canbus[:,0].unsqueeze(1))
            loss2 = criterion(out2, canbus[:,1].unsqueeze(1))
            loss = loss1 + loss2

            if args.tb != 'None':
                update_scalar_tb('training loss speed', loss1, epoch * len(trainloader) + i, args.tb)
                update_scalar_tb('training loss steering', loss2, epoch * len(trainloader) + i, args.tb)

            loss.backward()
            optimizer.step()

            rloss1 += loss1.item()
            rloss2 += loss2.item()

            # print every 100 groups
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f -- loss %.3f'
                     % (epoch + 1, i + 1, rloss1 / 100, rloss2 / 100))

                rloss1 = 0.0
                rloss2 = 0.0

        # Validation loss
        model.eval()

        correct_val_1 = 0
        correct_val_2 = 0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                images = data['image']
                canbus = data['can_bus']

                images = images.to(device)
                canbus = canbus.to(device)
                canbus = canbus.squeeze(0)


                out1, out2 = model(images)
                loss1_val = criterion(out1, canbus[:,0].unsqueeze(1))
                loss2_val = criterion(out2, canbus[:,1].unsqueeze(1))

                correct_val_1 += get_accuracy(out1, canbus[:,0], coef)
                correct_val_2 += get_accuracy(out2, canbus[:,1], coef)

                if args.tb != 'None':
                    update_scalar_tb('validation loss speed', loss1_val, epoch * len(valloader) + i, args.tb)
                    update_scalar_tb('validation loss steering', loss2_val, epoch * len(valloader) + i, args.tb)

            print('Val acc 1: %.4f -- Val acc 2: %.4f' % (correct_val_1/dataset_val.true_length(), correct_val_2/dataset_val.true_length()))

    print('Finished training')

# Save model
if args.save != 'None':
    torch.save(model.state_dict(), args.save)

print('Validating with %d groups of connected images' % (len(dataset_val)))

model.eval()
correct1 = 0
correct2 = 0

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
        canbus = data['can_bus']

        images = images.to(device)
        canbus = canbus.to(device)
        canbus = canbus.squeeze(0)

        out1, out2 = model(images)

        all_preds_1 = torch.cat((all_preds_1, out1[:,0]), dim=0)
        all_preds_2 = torch.cat((all_preds_2, out2[:,0]), dim=0)
        all_labels_1 = torch.cat((all_labels_1, canbus[:,0]), dim=0)
        all_labels_2 = torch.cat((all_labels_2, canbus[:,1]), dim=0)

        correct1 += get_accuracy(out1, canbus[:,0], coef)
        correct2 += get_accuracy(out2, canbus[:,1], coef)

    print('Acc 1: %.4f -- Val acc 2: %.4f' % (correct1/dataset_val.true_length(), correct2/dataset_val.true_length()))

# Unnormalize
for i in range(all_labels_1.shape[0]):
    all_labels_1[i] = all_labels_1[i] * std_sp + mean_sp
    all_preds_1[i] = all_preds_1[i] * std_sp + mean_sp
    all_labels_2[i] = all_labels_2[i] * std_st + mean_st
    all_preds_2[i] = all_preds_2[i] * std_st + mean_st

error1, max1, min1 = mean_squared_error(all_preds_1.cpu(), all_labels_1.cpu())
error2, max2, min2 = mean_squared_error(all_preds_2.cpu(), all_labels_2.cpu())
print('Mean squared error')
print('Speed: %.4f -- Steering %.4f' %(error1, error2))
print('Max error')
print('Speed: %.4f -- Steering %.4f' %(max1, max2))
print('Min error')
print('Speed: %.4f -- Steering %.4f' %(min1, min2))

draw_reg_lineplot(all_labels_1.cpu(), all_preds_1.cpu(), 'speed_reg_graf.png')
draw_reg_lineplot(all_labels_2.cpu(), all_preds_2.cpu(), 'steering_reg_graf.png')

show_predicted_data(valloader, None, None, all_preds_1.cpu(), all_preds_2.cpu(), mean, std, reg=True)
