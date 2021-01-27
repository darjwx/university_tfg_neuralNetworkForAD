# Utils
from utils.dataloader_hf import DataLoaderHF
from utils.transforms import Rescale, ToTensor, Normalize
from utils.metrics import get_metrics, show_predicted_data, update_scalar_tb, pr_curve_tb

# Misc
import time
import copy

# Pytorch
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# OpenCV
import cv2 as cv

#Numpy
import numpy as np

# Functions
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_1 = 0
            running_corrects_2 = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                images = data['image']
                labels = data['label']

                images = images.to(device)
                labels = labels.to(device)

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    out1, out2 = model(images)
                    loss1 = criterion(out1, labels[:, 0])
                    loss2 = criterion(out2, labels[:, 1])
                    loss = loss1 + loss2
                    _, predicted_1 = torch.max(out1.data, 1)
                    _, predicted_2 = torch.max(out2.data, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects_1 += (predicted_1 == labels[:, 0]).sum().item()
                running_corrects_2 += (predicted_2 == labels[:, 1]).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_1 = running_corrects_1 / len(dataloaders[phase].dataset)
            epoch_acc_2 = running_corrects_2 / len(dataloaders[phase].dataset)

            # TensorBoard
            if phase == 'train':
                update_scalar_tb('Train loss per epoch', epoch_loss, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}  {:.4f}'.format(phase, epoch_loss, epoch_acc_1, epoch_acc_2))

            # deep copy the model
            epoch_acc = (epoch_acc_1 + epoch_acc_2) / 2
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training and validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#Model
class MyModel(nn.Module):
    def __init__(self, num_classes_1, num_classes_2):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, num_classes_1)
        self.fc2 = nn.Linear(num_ftrs, num_classes_2)

    def forward(self, x):
        x = self.model(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2

# Parameters
num_epochs = 4
batch_size = 20
classes_1 = 3
classes_2 = 3
learning_rate = 0.001

# Transforms
# Original resolution / 4 (900, 1600) (h, w)
mean = (97.7419, 99.9757, 98.8718)
std = (56.8975, 55.1809, 55.8246)
composed = transforms.Compose([Rescale((225,400)),
                              ToTensor(),
                              Normalize(mean, std)])

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(classes_1, classes_2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Custom Dataloader for NuScenes
HOME_ROUTE = '/media/darjwx/ssd_data/data/sets/nuscenes/'
dataset_train = DataLoaderHF(HOME_ROUTE, 'train', 1111, 850, composed)
dataset_val = DataLoaderHF(HOME_ROUTE, 'val', 1111, 850, composed)

classes_speed = ['stop', 'stoping', 'accel']
classes_steering = ['straight', 'left', 'right']

trainloader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4)
valloader = DataLoader(dataset_val, batch_size, shuffle=True, num_workers=4)

dataloaders = {'train': trainloader, 'val': valloader}
model = train_model(model, dataloaders, criterion, optimizer, num_epochs)

#Statistics
print('---Statistics---')
class_correct_1 = list(0. for i in range(3))
class_total_1 = list(0. for i in range(3))
class_correct_2 = list(0. for i in range(3))
class_total_2 = list(0. for i in range(3))

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


# p-r curve
preds_1 = torch.cat([torch.stack(batch) for batch in preds_1])
preds_2 = torch.cat([torch.stack(batch) for batch in preds_2])

pr_curve_tb(3, all_labels_1, all_labels_2, preds_1, preds_2)

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

#Show examples
show_predicted_data(valloader, classes_speed, classes_steering, all_preds_1.cpu(), all_preds_2.cpu(), mean, std)
