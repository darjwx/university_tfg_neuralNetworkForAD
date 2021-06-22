# Utils and dataloader
from utils.dataloader_seq import DataLoaderSeq
from utils.transforms import Rescale, ToTensor, Normalize

# Metrics
from utils.metrics import get_metrics, show_predicted_data, update_scalar_tb, pr_curve_tb, dummy_classifier, draw_lineplot

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
parser.add_argument('--layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--res', nargs=2, type=int, default=[225,400], help='Images resolution')
parser.add_argument('--weights', nargs=3, type=float, default=[1., 1., 1.], help='Loss weights')
parser.add_argument('--canbus', type=str_to_bool, default=False, help='Wheter to use canbus data as an input')
parser.add_argument('--route', type=str, default='/data/sets/nuscenes/', help='Route where the NuScenes dataset is located')
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
num_classes = 3
if args.canbus:
    add_dim = 2
else:
    add_dim = 0

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (0.3833, 0.3921, 0.3877)
std = (0.2231, 0.2164, 0.2189)
composed = transforms.Compose([Rescale(tuple(args.res), canbus=args.canbus, seq=True),
                              ToTensor(canbus=args.canbus, seq=True),
                              Normalize(mean, std, canbus=args.canbus, seq=True)])

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
        self.fc2 = nn.Linear(120 + add_dim, 84)

        # LSTM and output linear layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, num_classes)


    def forward(self, x, d=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        batch_size, sl, C, H, W = x.size()
        x = x.view(batch_size * sl, C, H, W)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 53 * 97)
        x = F.relu(self.fc1(x))

        if args.canbus:
            d = d.squeeze(0)
            x = torch.cat((x, d), dim=1)

        x = F.relu(self.fc2(x))
        x = x.unsqueeze(0)
        x, _ = self.lstm(x, (h0, c0))
        x = x.view(-1, hidden_size)
        out1 = self.fc3(x)
        out2 = self.fc4(x)

        return out1, out2


# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNtoLSTM(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

weights = torch.from_numpy(np.array(args.weights)).float().to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
# scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 2, verbose = True)

# Custom Dataloader for NuScenes
HOME_ROUTE = args.route
dataset_train = DataLoaderSeq(HOME_ROUTE, 'train', 1111, 850, composed, args.canbus)
dataset_val = DataLoaderSeq(HOME_ROUTE, 'val', 1111, 850, composed, args.canbus)

classes_speed = ['maintain', 'stoping', 'accel']

classes_steering = ['straight', 'left', 'right']

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
            labels = data['label']
            images = images.to(device)
            labels = labels.to(device)

            if args.canbus:
                ndata = data['numerical']
                ndata = ndata.to(device)

                # Use previous data
                ndata = torch.roll(ndata, 1, dims=1)
                ndata[0, 0, 0] = 0.0
                ndata[0, 0, 1] = 0.0

                out1, out2 = model(images, ndata)
            else:
                out1, out2 = model(images)

            labels = labels.view(-1, 2)
            loss1 = criterion(out1, labels[:, 0])
            loss2 = criterion(out2, labels[:, 1])
            loss = loss1 + loss2

            if args.tb != 'None':
                update_scalar_tb('training loss speed', loss1, epoch * len(trainloader) + i, args.tb)
                update_scalar_tb('training loss direction', loss2, epoch * len(trainloader) + i, args.tb)

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
        model.eval()

        correct_val_1 = 0
        correct_val_2 = 0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                images = data['image']
                labels = data['label']
                images = images.to(device)
                labels = labels.to(device)

                if args.canbus:
                    ndata = data['numerical']
                    ndata = ndata.to(device)

                    # Use previous data
                    ndata = torch.roll(ndata, 1, dims=1)
                    ndata[0, 0, 0] = 0.0
                    ndata[0, 0, 1] = 0.0

                    out1, out2 = model(images, ndata)
                else:
                    out1, out2 = model(images)

                labels = labels.view(-1, 2)
                loss1_val = criterion(out1, labels[:, 0])
                loss2_val = criterion(out2, labels[:, 1])

                _, predicted_1 = torch.max(out1.data, 1)
                _, predicted_2 = torch.max(out2.data, 1)
                correct_val_1 += (predicted_1 == labels[:, 0]).sum().item()
                correct_val_2 += (predicted_2 == labels[:, 1]).sum().item()

                if args.tb != 'None':
                    update_scalar_tb('validation loss speed', loss1_val, epoch * len(valloader) + i, args.tb)
                    update_scalar_tb('validation loss direction', loss2_val, epoch * len(valloader) + i, args.tb)

            print('Val acc 1: %.4f --- Val acc 2: %.4f' % (correct_val_1/dataset_val.true_length(), correct_val_2/dataset_val.true_length()))

    print('Finished training')

# Save model
if args.save != 'None':
    torch.save(model.state_dict(), args.save)

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

model.eval()
with torch.no_grad():
    for v, data in enumerate(valloader):
        images = data['image']
        labels = data['label']
        images = images.to(device)
        labels = labels.to(device)

        if args.canbus:
            ndata = data['numerical']
            ndata = ndata.to(device)

            # Use previous data
            ndata = torch.roll(ndata, 1, dims=1)
            ndata[0, 0, 0] = 0.0
            ndata[0, 0, 1] = 0.0

            out1, out2 = model(images, ndata)
        else:
            out1, out2 = model(images)

        labels = labels.view(-1, 2)
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

accuracy_t_1 = correct_total_1 / dataset_val.true_length()
accuracy_t_2 = correct_total_2 / dataset_val.true_length()
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

# Plot commands vs time
draw_lineplot(all_labels_1.cpu(), all_preds_1.cpu(), classes_speed)
draw_lineplot(all_labels_2.cpu(), all_preds_2.cpu(), classes_steering)

# p-r curve
preds_1 = torch.cat([torch.stack(batch) for batch in preds_1])
preds_2 = torch.cat([torch.stack(batch) for batch in preds_2])

if args.tb != 'None':
    pr_curve_tb(3, all_labels_1, all_labels_2, preds_1, preds_2, args.tb)

# Recall, precision, f1_score and confusion_matrix
get_metrics(all_labels_1.cpu(), all_preds_1.cpu(), 3, classes_speed)
get_metrics(all_labels_2.cpu(), all_preds_2.cpu(), 3, classes_steering)

show_predicted_data(valloader, classes_speed, classes_steering, all_preds_1.cpu(), all_preds_2.cpu(), mean, std, True)
