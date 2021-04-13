# Utils and dataloader
from utils.dataloader_areg import DataLoaderAReg
from utils.transforms import Rescale, ToTensor, Normalize

# Metrics
from utils.metrics import show_predicted_data, update_scalar_tb, draw_reg_lineplot, mean_squared_error

# Pytorch
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

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
parser.add_argument('--lw', type=int, default=1, help='Loss weights')
parser.add_argument('--video_name', type=str, default='val_info_custom.avi', help='Output video name')
parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--video', type=str_to_bool, default=False, help='Wheter to build a video')

args = parser.parse_args()

# Parameters
num_layers = args.layers
hidden_size = 128
num_epochs = args.epochs
batch_size = 1
learning_rate = args.lr
output = 1
classes = 3
coef = 0.15
lw = args.lw
video = args.video_name

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

mean_sp = 19.2159
std_sp = 3.0407
mean_st0 = -0.6561
std_st0 = 20.5480
mean_st1 = -54.7190
std_st1 = 174.4027
mean_st2 = 23.4976
std_st2 = 174.7414

composed = transforms.Compose([Rescale((225,400), areg=True),
                              ToTensor(areg=True),
                              Normalize(mean, std, mean_sp, std_sp, mean_st0, std_st0, mean_st1, std_st1, mean_st2, std_st2, areg=True)])

class AidedRegression(nn.Module):
    def __init__(self, hidden_size, num_layers, output):
        super(AidedRegression, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Conv layers: Resnet
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # LSTM
        self.lstm = nn.LSTM(num_ftrs, hidden_size, num_layers, batch_first=True)

        # Regression: speed -> 1
        self.fc3 = nn.Linear(hidden_size, output)

        # Steering type classification -> 3
        self.fc4 = nn.Linear(hidden_size, classes)
        # Regression: steering -> 1
        self.fc5 = nn.Linear(hidden_size, 3)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        batch_size, sl, C, H, W = x.size()
        x = x.view(batch_size * sl, C, H, W)
        x = self.resnet(x)

        x = x.unsqueeze(0)
        x, _ = self.lstm(x, (h0, c0))

        x = x.view(-1, hidden_size)

        out1 = self.fc3(x)

        type = self.fc4(x)
        out2 = self.fc5(x)

        return out1, out2, type



# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AidedRegression(hidden_size, num_layers, output)
model = model.to(device)

criterion_reg = nn.L1Loss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Custom Dataloader for NuScenes
HOME_ROUTE = '/media/darjwx/ssd_data/data/sets/nuscenes/'
dataset_train = DataLoaderAReg(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderAReg(HOME_ROUTE, 'val', 1111, 850, composed)

trainloader = DataLoader(dataset_train, batch_size, shuffle=True)
valloader = DataLoader(dataset_val, batch_size, shuffle=False)

print('Training with %d groups of connected images' % (len(dataset_train)))

for epoch in range(num_epochs):
    rloss1 = 0.0
    rloss2 = 0.0
    rloss3 = 0.0

    model.train()
    for i, data in enumerate(trainloader):

        model.zero_grad()
        images = data['image']
        speed = data['speed']
        steering_type = data['s_type']
        steering = data['steering']

        images = images.to(device)
        speed = speed.to(device).squeeze(0)
        steering_type = steering_type.to(device).squeeze(0)
        steering = steering.to(device).squeeze(0)

        out1, out2, type = model(images)

        # canbus[:,0] -> (42,) -- unsqueeze(1) -> (42,1)
        loss1 = criterion_reg(out1, speed.unsqueeze(1))
        loss2 = criterion_class(type, steering_type)

        # List slicing to calculate the loss with the correct values
        # Rows index: idx -- Columns index: steering_type
        # [*, 1, 1]
        # [1, *, 1]
        # [1, *, 1]
        # [1, 1, *]
        idx = torch.arange(out2.size(0))
        loss3 = criterion_reg(out2[idx,steering_type], steering)
        loss = loss1 + lw*loss2 + lw*loss3

        update_scalar_tb('Train loss: Speed regression', loss1, epoch * len(trainloader) + i)
        update_scalar_tb('Train loss: Steering classification', loss2, epoch * len(trainloader) + i)
        update_scalar_tb('Train loss: Steering regression', loss3, epoch * len(trainloader) + i)

        loss.backward()
        optimizer.step()

        rloss1 += loss1.item()
        rloss2 += loss2.item()
        rloss3 += loss3.item()

        # print every 100 groups
        if i % 100 == 99:
            print('[%d, %5d] Speed reg loss: %.3f -- Class loss %.3f -- Steering reg loss %.3f'
                 % (epoch + 1, i + 1, rloss1 / 100, rloss2 / 100, rloss3 / 100))

            rloss1 = 0.0
            rloss2 = 0.0
            rloss3 = 0.0

    # Validation loss
    model.eval()

    correct_val_1 = 0
    correct_val_2 = 0
    with torch.no_grad():
        for i, data in enumerate(valloader):
            images = data['image']
            speed = data['speed']
            steering_type = data['s_type']
            steering = data['steering']

            images = images.to(device)
            speed = speed.to(device).squeeze(0)
            steering_type = steering_type.to(device).squeeze(0)
            steering = steering.to(device).squeeze(0)


            out1, out2, type = model(images)
            loss1 = criterion_reg(out1, speed.unsqueeze(1))
            loss2 = criterion_class(type, steering_type)

            idx = torch.arange(out2.size(0))
            loss3 = criterion_reg(out2[idx,steering_type], steering)

            update_scalar_tb('Val loss: Speed regression', loss1, epoch * len(valloader) + i)
            update_scalar_tb('Val loss: Steering classification', loss2, epoch * len(valloader) + i)
            update_scalar_tb('Val loss: Steering regression', loss3, epoch * len(valloader) + i)

print('Finished training')

print('Validating with %d groups of connected images' % (len(dataset_val)))

model.eval()
correct1 = 0
correct2 = 0

all_preds_1 = torch.tensor([])
all_preds_2 = torch.tensor([])
all_labels_1 = torch.tensor([])
all_labels_2 = torch.tensor([])
all_type = torch.tensor([])
pred_type = torch.tensor([])
all_preds_1 = all_preds_1.to(device)
all_preds_2 = all_preds_2.to(device)
all_labels_1 = all_labels_1.to(device)
all_labels_2 = all_labels_2.to(device)
all_type = all_type.to(device)
pred_type = pred_type.to(device)

with torch.no_grad():
    for v, data in enumerate(valloader):

        images = data['image']
        speed = data['speed']
        steering_type = data['s_type']
        steering = data['steering']

        images = images.to(device)
        speed = speed.to(device).squeeze(0)
        steering_type = steering_type.to(device).squeeze(0)
        steering = steering.to(device).squeeze(0)

        out1, out2, type = model(images)
        _, predicted = torch.max(type.data, 1)
        idx = torch.arange(out2.size(0))

        all_preds_1 = torch.cat((all_preds_1, out1[:,0]), dim=0)
        all_preds_2 = torch.cat((all_preds_2, out2[idx,steering_type]), dim=0)
        all_labels_1 = torch.cat((all_labels_1, speed), dim=0)
        all_labels_2 = torch.cat((all_labels_2, steering), dim=0)
        all_type = torch.cat((all_type, steering_type), dim=0)
        pred_type = torch.cat((pred_type, predicted), dim=0)

# Unnormalize
for i in range(all_labels_1.shape[0]):
    all_labels_1[i] = all_labels_1[i] * std_sp + mean_sp
    all_preds_1[i] = all_preds_1[i] * std_sp + mean_sp

    if all_type[i] == 0:
        all_labels_2[i] = all_labels_2[i] * std_st0 + mean_st0
        all_preds_2[i] = all_preds_2[i] * std_st0 + mean_st0
    elif all_type[i] == 1:
        all_labels_2[i] = all_labels_2[i] * std_st1 + mean_st1
        all_preds_2[i] = all_preds_2[i] * std_st1 + mean_st1
    else:
        all_labels_2[i] = all_labels_2[i] * std_st2 + mean_st2
        all_preds_2[i] = all_preds_2[i] * std_st2 + mean_st2

if args.video:
    dataset_val.create_video(video, pred_type.cpu(), all_preds_2.cpu(), all_labels_2.cpu())

# Mean squared error
error1, max1, min1 = mean_squared_error(all_preds_1.cpu(), all_labels_1.cpu())
error2, max2, min2 = mean_squared_error(all_preds_2.cpu(), all_labels_2.cpu())
print('Mean squared error')
print('Speed: %.4f -- Steering %.4f' %(error1, error2))
print('Max error')
print('Speed: %.4f -- Steering %.4f' %(max1, max2))
print('Min error')
print('Speed: %.4f -- Steering %.4f' %(min1, min2))

draw_reg_lineplot(all_labels_1.cpu(), all_preds_1.cpu())
draw_reg_lineplot(all_labels_2.cpu(), all_preds_2.cpu())

#show_predicted_data(valloader, None, None, all_preds_1.cpu(), all_preds_2.cpu(), mean, std, reg=True)
