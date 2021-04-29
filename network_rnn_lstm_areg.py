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
parser.add_argument('--hidden', type=int, default=128, help='LSTM hidden size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lw', type=int, default=1, help='Loss weights')
parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--predf', type=str_to_bool, default=False, help='Wheter to use predictions to filter the regression targets')
parser.add_argument('--route', type=str, default='/data/sets/nuscenes/', help='Route where the NuScenes dataset is located')
parser.add_argument('--res', nargs=2, type=int, default=[225,400], help='Images resolution')
parser.add_argument('--tb', type=str_to_bool, default=False, help='Wheter to upload data to TensorBoard')
parser.add_argument('--weights_sp', nargs=2, type=float, default=[1., 1.], help='Loss weights for speed')
parser.add_argument('--weights_st', nargs=3, type=float, default=[1., 1., 1.], help='Loss weights for steering')
parser.add_argument('--save', type=str, default='None', help='Location where the model is going to be saved')
parser.add_argument('--load', type=str, default='None', help='Path to the model to be loaded')
parser.add_argument('--video', type=str, default='None', help='Path for the output video')

args = parser.parse_args()

# Parameters
num_layers = args.layers
hidden_size = args.hidden
num_epochs = args.epochs
batch_size = 1
learning_rate = args.lr
out_sp = 2
out_st = 3
lw = args.lw

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

mean_sp = 21.9203
std_sp = 10.6167
mean_st0 = -0.6281
std_st0 = 20.1580
mean_st1 = -167.8248
std_st1 = 74.4673
mean_st2 = 160.7079
std_st2 = 66.5891

composed = transforms.Compose([Rescale(tuple(args.res), areg=True),
                              ToTensor(areg=True),
                              Normalize(mean, std, mean_sp, std_sp, mean_st0, std_st0, mean_st1, std_st1, mean_st2, std_st2, areg=True)])

class AidedRegression(nn.Module):
    def __init__(self, hidden_size, num_layers, out_sp, out_st):
        super(AidedRegression, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Conv layers: Resnet
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # LSTM
        self.lstm = nn.LSTM(num_ftrs, hidden_size, num_layers, batch_first=True)

        # Speed type classification -> 2
        self.fc2 = nn.Linear(hidden_size, out_sp)
        # Regression: speed -> 1
        self.fc3 = nn.Linear(hidden_size, out_sp)

        # Steering type classification -> 3
        self.fc4 = nn.Linear(hidden_size, out_st)
        # Regression: steering -> 1
        self.fc5 = nn.Linear(hidden_size, out_st)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        batch_size, sl, C, H, W = x.size()
        x = x.view(batch_size * sl, C, H, W)
        x = self.resnet(x)

        x = x.unsqueeze(0)
        x, _ = self.lstm(x, (h0, c0))

        x = x.view(-1, hidden_size)

        type_sp = self.fc2(x)
        out1 = self.fc3(x)

        type_st = self.fc4(x)
        out2 = self.fc5(x)

        return type_sp, out1, type_st, out2



# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AidedRegression(hidden_size, num_layers, out_sp, out_st)
model = model.to(device)

weights_sp = torch.from_numpy(np.array(args.weights_sp)).float().to(device)
weights_st = torch.from_numpy(np.array(args.weights_st)).float().to(device)

criterion_class_sp = nn.CrossEntropyLoss(weight=weights_sp)
criterion_class_st = nn.CrossEntropyLoss(weight=weights_st)
criterion_reg = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Custom Dataloader for NuScenes
HOME_ROUTE = args.route
dataset_train = DataLoaderAReg(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderAReg(HOME_ROUTE, 'val', 1111, 850, composed)

trainloader = DataLoader(dataset_train, batch_size, shuffle=True)
valloader = DataLoader(dataset_val, batch_size, shuffle=False)

print('Training with %d groups of connected images' % (len(dataset_train)))

if args.load != 'None':
    print('Loading model from %s' % (args.load))
    model.load_state_dict(torch.load(args.load))
else:
    for epoch in range(num_epochs):
        rloss1 = 0.0
        rloss2 = 0.0
        rloss3 = 0.0
        rloss4 = 0.0

        model.train()
        for i, data in enumerate(trainloader):

            model.zero_grad()
            images = data['image']
            speed_type = data['sp_type']
            speed = data['speed']
            steering_type = data['st_type']
            steering = data['steering']

            images = images.to(device)
            speed_type = speed_type.to(device).squeeze(0)
            speed = speed.to(device).squeeze(0)
            steering_type = steering_type.to(device).squeeze(0)
            steering = steering.to(device).squeeze(0)

            type_sp, out1, type_st, out2 = model(images)

            _, predicted_sp = torch.max(type_sp.data, 1)
            _, predicted_st = torch.max(type_st.data, 1)

            loss1 = criterion_class_sp(type_sp, speed_type)
            loss3 = criterion_class_st(type_st, steering_type)

            # List slicing to calculate the loss with the correct values
            # Rows index: idx -- Columns index: speed_type/steering_type
            # [*, 1, 1]
            # [1, *, 1]
            # [1, *, 1]
            # [1, 1, *]
            idx_sp = torch.arange(out1.size(0))
            idx_st = torch.arange(out2.size(0))

            if args.predf:
                loss2 = criterion_reg(out1[idx_sp,predicted_sp], speed)
                loss4 = criterion_reg(out2[idx_st,predicted_st], steering)
            else:
                loss2 = criterion_reg(out1[idx_sp,speed_type], speed)
                loss4 = criterion_reg(out2[idx_st,steering_type], steering)

            loss = lw*loss1 + lw*loss2 + lw*loss3 + lw*loss4

            if args.tb:
                update_scalar_tb('Train loss: Speed classification', loss1, epoch * len(trainloader) + i)
                update_scalar_tb('Train loss: Speed regression', loss2, epoch * len(trainloader) + i)
                update_scalar_tb('Train loss: Steering classification', loss3, epoch * len(trainloader) + i)
                update_scalar_tb('Train loss: Steering regression', loss4, epoch * len(trainloader) + i)

            loss.backward()
            optimizer.step()

            rloss1 += loss1.item()
            rloss2 += loss2.item()
            rloss3 += loss3.item()
            rloss4 += loss4.item()

            # print every 100 groups
            if i % 100 == 99:
                print('[%d, %5d] Speed class loss: %.3f -- Speed reg loss: %.3f -- Steering class loss %.3f -- Steering reg loss %.3f'
                     % (epoch + 1, i + 1, rloss1 / 100, rloss2 / 100, rloss3 / 100, rloss4 / 100))

                rloss1 = 0.0
                rloss2 = 0.0
                rloss3 = 0.0
                rloss4 = 0.0

        # Validation loss
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(valloader):
                images = data['image']
                speed_type = data['sp_type']
                speed = data['speed']
                steering_type = data['st_type']
                steering = data['steering']

                images = images.to(device)
                speed_type = speed_type.to(device).squeeze(0)
                speed = speed.to(device).squeeze(0)
                steering_type = steering_type.to(device).squeeze(0)
                steering = steering.to(device).squeeze(0)


                type_sp, out1, type_st, out2 = model(images)

                _, predicted_sp = torch.max(type_sp.data, 1)
                _, predicted_st = torch.max(type_st.data, 1)

                loss1 = criterion_class_sp(type_sp, speed_type)
                loss3 = criterion_class_st(type_st, steering_type)

                idx_sp = torch.arange(out1.size(0))
                idx_st = torch.arange(out2.size(0))

                if args.predf:
                    loss2 = criterion_reg(out1[idx_sp,predicted_sp], speed)
                    loss4 = criterion_reg(out2[idx_st,predicted_st], steering)
                else:
                    loss2 = criterion_reg(out1[idx_sp,speed_type], speed)
                    loss4 = criterion_reg(out2[idx_st,steering_type], steering)

                if args.tb:
                    update_scalar_tb('Val loss: Speed classification', loss1, epoch * len(valloader) + i)
                    update_scalar_tb('Val loss: Speed regression', loss2, epoch * len(valloader) + i)
                    update_scalar_tb('Val loss: Steering classification', loss3, epoch * len(valloader) + i)
                    update_scalar_tb('Val loss: Steering regression', loss4, epoch * len(valloader) + i)

print('Finished training')

# Save model
if args.save != 'None':
    torch.save(model.state_dict(), args.save)

print('Validating with %d groups of connected images' % (len(dataset_val)))

model.eval()
correct1 = 0
correct2 = 0

speed_labels_gt = torch.tensor([])
steering_labels_gt = torch.tensor([])
speed_reg_gt = torch.tensor([])
steering_reg_gt = torch.tensor([])
speed_labels_pred = torch.tensor([])
steering_labels_pred = torch.tensor([])
speed_reg_pred = torch.tensor([])
steering_reg_pred = torch.tensor([])

speed_labels_gt = speed_labels_gt.to(device)
steering_labels_gt = steering_labels_gt.to(device)
speed_reg_gt = speed_reg_gt.to(device)
steering_reg_gt = steering_reg_gt.to(device)
speed_labels_pred = speed_labels_pred.to(device)
steering_labels_pred = steering_labels_pred.to(device)
speed_reg_pred = speed_reg_pred.to(device)
steering_reg_pred = steering_reg_pred.to(device)

with torch.no_grad():
    for v, data in enumerate(valloader):

        images = data['image']
        speed_type = data['sp_type']
        speed = data['speed']
        steering_type = data['st_type']
        steering = data['steering']

        images = images.to(device)
        speed_type = speed_type.to(device).squeeze(0)
        speed = speed.to(device).squeeze(0)
        steering_type = steering_type.to(device).squeeze(0)
        steering = steering.to(device).squeeze(0)

        type_sp, out1, type_st, out2 = model(images)
        _, predicted_sp = torch.max(type_sp.data, 1)
        _, predicted_st = torch.max(type_st.data, 1)

        idx_sp = torch.arange(out1.size(0))
        idx_st = torch.arange(out2.size(0))

        if args.predf:
            speed_reg_pred = torch.cat((speed_reg_pred, out1[idx_sp,predicted_sp]), dim=0)
            steering_reg_pred = torch.cat((steering_reg_pred, out2[idx_st,predicted_st]), dim=0)
        else:
            speed_reg_pred = torch.cat((speed_reg_pred, out1[idx_sp,speed_type]), dim=0)
            steering_reg_pred = torch.cat((steering_reg_pred, out2[idx_st,steering_type]), dim=0)

        speed_labels_gt = torch.cat((speed_labels_gt, speed_type), dim=0)
        steering_labels_gt = torch.cat((steering_labels_gt, steering_type), dim=0)
        speed_reg_gt = torch.cat((speed_reg_gt, speed), dim=0)
        steering_reg_gt = torch.cat((steering_reg_gt, steering), dim=0)
        speed_labels_pred = torch.cat((speed_labels_pred, predicted_sp), dim=0)
        steering_labels_pred = torch.cat((steering_labels_pred, predicted_st), dim=0)

# Unnormalize
if args.predf:
    aux_speed = speed_labels_pred
    aux_steering = steering_labels_pred
else:
    aux_speed = speed_labels_gt
    aux_steering = steering_labels_gt

for i in range(speed_reg_gt.shape[0]):
    if speed_labels_gt[i] == 1:
        speed_reg_gt[i] = speed_reg_gt[i] * std_sp + mean_sp
    if aux_speed[i] == 1:
        speed_reg_pred[i] = speed_reg_pred[i] * std_sp + mean_sp

    if steering_labels_gt[i] == 0:
        steering_reg_gt[i] = steering_reg_gt[i] * std_st0 + mean_st0
    elif steering_labels_gt[i] == 1:
        steering_reg_gt[i] = steering_reg_gt[i] * std_st1 + mean_st1
    else:
        steering_reg_gt[i] = steering_reg_gt[i] * std_st2 + mean_st2
    if aux_steering[i] == 0:
        steering_reg_pred[i] = steering_reg_pred[i] * std_st0 + mean_st0
    elif aux_steering[i] == 1:
        steering_reg_pred[i] = steering_reg_pred[i] * std_st1 + mean_st1
    else:
        steering_reg_pred[i] = steering_reg_pred[i] * std_st2 + mean_st2

if args.video != 'None':
    dataset_val.create_video(args.video, speed_labels_pred.cpu(), steering_labels_pred.cpu(), speed_reg_pred.cpu(), steering_reg_pred.cpu())

# Mean squared error
error1, max1, min1 = mean_squared_error(speed_reg_pred.cpu(), speed_reg_gt.cpu())
error2, max2, min2 = mean_squared_error(steering_reg_pred.cpu(), steering_reg_gt.cpu())
print('Mean squared error')
print('Speed: %.4f -- Steering %.4f' %(error1, error2))
print('Max error')
print('Speed: %.4f -- Steering %.4f' %(max1, max2))
print('Min error')
print('Speed: %.4f -- Steering %.4f' %(min1, min2))

draw_reg_lineplot(speed_reg_gt.cpu(), speed_reg_pred.cpu(), 'speed_areg_graf.png')
draw_reg_lineplot(steering_reg_gt.cpu(), steering_reg_pred.cpu(), 'steering_areg_graf.png')

#show_predicted_data(valloader, None, None, speed_reg_pred.cpu(), steering_reg_pred.cpu(), mean, std, reg=True)
