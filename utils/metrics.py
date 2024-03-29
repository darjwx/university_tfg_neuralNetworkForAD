"""
    Metrics - Different functions to build the metrics used across all the models.
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

# sklearn metrics
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Pytorch
from torchvision.utils import make_grid

from statistics import mode

# Numpy, matplotlib and pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Seaborn
import seaborn as sns
sns.set_theme(style='darkgrid', palette='pastel')

# TensorBoard
from torch.utils.tensorboard import SummaryWriter


def get_metrics(labels_true, labels_pred, num_classes, classes):
    """
    Uses sklearn to calculate different metrics:
     - Recall
     - Precision
     - F1 score
     - Confusion matrix
    :param labels_true: Ground truth.
    :param labels_pred: Predicted labels.
    :param num_classes: Number of classes.
    :param classes: List of classes.
    """

    # Matplotlib theme
    plt.style.use('classic')

    # Metrics
    recall = recall_score(labels_true, labels_pred, average=None, zero_division=0)
    precision = precision_score(labels_true, labels_pred, average=None, zero_division=0)
    f1 = f1_score(labels_true, labels_pred, average=None, zero_division=0)
    conf_matrix = confusion_matrix(labels_true, labels_pred)

    # Print metrics
    for i in range(num_classes):
        print('Recall, precision and F1-score of %5s: %1.3f %1.3f %1.3f'
              % (classes[i], recall[i], precision[i], f1[i]))

    # Display the conf matrix
    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = classes)
    disp.plot()
    plt.show()


def show_predicted_data(dataloader, classes_1, classes_2, labels_pred_1, labels_pred_2, mean, std, seq=False, reg=False):
    """
    Shows a batch of images with their corresponding prediction and ground truth.
    :param dataloader: Dataloader variable from pytorch with images and labels.
    :param classes_1: Number of classes 1.
    :param classes_2: Number of classes 2.
    :param labels_pred_1: Predictions 1.
    :param labels_pred_2: Predictions 2.
    :param mean: Mean of each channel.
    :param std: Standard deviation of each channel.
    """

    # Matplotlib theme
    plt.style.use('classic')

    # Adapts the image for plt.imshow
    def imshow(image):
        img = image
        # Unnormalize images
        for i in range(3):
            img[i] = img[i] * std[i] + mean[i]

        img = img.numpy()
        # imshow expects and array with the form: H * W * Colour channels
        img = np.transpose(img, (1, 2, 0))
        # BGR -> RGB
        plt.imshow(img[...,::-1])
        plt.show()

    labels_pred_1 = labels_pred_1.numpy().astype(np.uint8)
    labels_pred_2 = labels_pred_2.numpy().astype(np.uint8)

    for i, data in enumerate(dataloader):

        images = data['image']
        if not reg:
            labels = data['label']
        else:
            canbus = data['can_bus']
            canbus = canbus.squeeze(0)
            batch_size, sl, C, H, W = images.size()
            images = images.view(batch_size * sl, C, H, W)


        if seq:
            labels = labels.view(-1, 2)

            batch_size, sl, C, H, W = images.size()
            images = images.view(batch_size * sl, C, H, W)

        if not reg:
            labels = labels.numpy()

            print('------')
            for j in range(np.shape(labels)[0]):
                info = 'predicted: {} gt: {}'.format(classes_1[labels_pred_1[j]], classes_1[labels[j,0]])
                print(str(j), 'Speed: ', info)

                info = 'predicted: {} gt: {}'.format(classes_2[labels_pred_2[j]], classes_2[labels[j,1]])
                print('  Steering: ', info)
            print('------')

        else:
            print('------')
            for j in range(np.shape(canbus)[0]):
                info = 'predicted: {} gt: {}'.format(labels_pred_1[j], canbus[j,0])
                print(str(j), 'Speed: ', info)

                info = 'predicted: {} gt: {}'.format(labels_pred_2[j], canbus[j,1])
                print('  Steering: ', info)
            print('------')


        imshow(make_grid(images))

def draw_lineplot(labels, preds, classes):
    """
    Builds a commands versus time lineplot.
    :param labels: Ground truth labels.
    :param preds: Predicted labels.
    :param classes: Name of each class.
    """

    type1 = np.empty(np.shape(labels)[0], dtype=np.dtype('<U122'))
    type2 = np.empty(np.shape(labels)[0], dtype=np.dtype('<U122'))
    lb = np.empty(np.shape(labels)[0], dtype=np.dtype('<U122'))
    pr = np.empty(np.shape(labels)[0], dtype=np.dtype('<U122'))
    time = np.empty(np.shape(labels)[0])

    # Build the data vectors
    labels = labels.numpy().astype(np.uint8)
    preds = preds.numpy().astype(np.uint8)
    for i in range(np.shape(labels)[0]):
        lb[i] = classes[labels[i]]
        type1[i] = 'gt'
        pr[i] = classes[preds[i]]
        type2[i] = 'preds'
        time[i] = i / 10

    commands = np.concatenate((lb, pr), 0)
    type = np.concatenate((type1, type2), 0)
    time = np.concatenate((time, time), 0)

    # Dataframe
    d = {'time': time,
         'commands': commands,
         'type': type}
    df = pd.DataFrame(data=d)

    # Draw plot
    sns.lineplot(x='time', y='commands', hue='type', data=df)

def draw_reg_lineplot(labels, preds, file, show=False):

    type1 = np.empty(np.shape(labels)[0], dtype=np.dtype('<U122'))
    type2 = np.empty(np.shape(labels)[0], dtype=np.dtype('<U122'))
    lb = np.empty(np.shape(labels)[0], dtype=np.dtype(np.float32))
    pr = np.empty(np.shape(labels)[0], dtype=np.dtype(np.float32))
    time = np.empty(np.shape(labels)[0])

    # Build the data vectors
    labels = labels.numpy()
    preds = preds.numpy()
    for i in range(np.shape(labels)[0]):
        type1[i] = 'gt'
        type2[i] = 'preds'
        time[i] = i / 10

    commands = np.concatenate((labels, preds), 0)
    type = np.concatenate((type1, type2), 0)
    time = np.concatenate((time, time), 0)

    # Dataframe
    d = {'time': time,
         'commands': commands,
         'type': type}
    df = pd.DataFrame(data=d)

    # Draw plot
    sns.lineplot(x='time', y='commands', hue='type', data=df)
    plt.savefig(file, dpi=100)
    if show:
        plt.show()

def update_scalar_tb(tag, scalar, x, route):
    """
    Writes a scalar value to TensorBoard
    :param tag: Name of the graph.
    :param scalar: Scalar to write.
    :param x: X axis.
    :param route: logs route.
    """

    writer = SummaryWriter(route)
    writer.add_scalar(tag, scalar, x)
    writer.close()

def pr_curve_tb(num_classes, labels_1, labels_2, preds_1, preds_2, route):
    """
    Writes a per class precision-recall curve to TensorBoard.
    :param num_classes: Number of classes.
    :param labels_1: List of labels 1.
    :param labels_2: List of labels 2.
    :param preds_1: List of predictions 1.
    :param preds_2: List of predictions 2.
    :param route: logs route.
    """
    writer = SummaryWriter(route)
    for i in range(num_classes):
        labels_i_1 = labels_1 == i
        labels_i_2 = labels_2 == i

        writer.add_pr_curve('Precision-Recall speed' + str(i), labels_i_1, preds_1[:,i], global_step=0)
        writer.add_pr_curve('Precision-Recall direction' + str(i), labels_i_2, preds_2[:,i], global_step=0)
        writer.close()

def visualize_model(net, images, route):
    """
    Visualise the model in TensorBoard.
    :param net: Neural network model.
    :param images: Input of the model.
    :param route: logs route.
    """

    writer = SummaryWriter(route)
    writer.add_graph(net, images)
    writer.close()

def dummy_classifier(ground_truth, const = 0):
    """
    Creates a dummy classifier that always predicts the same value.
    :param ground_truth: Ground truth labels.
    :param const: value to be predicted.
    """

    stry = ['most_frequent', 'constant']
    ground_truth = ground_truth.numpy().astype(np.uint8)

    for i in stry:
        if i == 'most_frequent':
            mf = mode(ground_truth)
            pred = np.full_like(ground_truth, mf)
        elif i == 'constant':
            pred = np.full_like(ground_truth, const)

        aux = 0
        for j in range(np.shape(ground_truth)[0]):
            if pred[j] == ground_truth[j]:
                aux += 1
        score = (aux / np.shape(ground_truth)[0]) * 100

        print('--------------------')
        print('Strategy: %s' %(i))
        print('Accuracy score of the dummy classifier: %.4f' %(score))

# Regression metrics
def get_accuracy(predicted, ground_truth, coef):
    """
    Calculates accuracy with a confidence coef.
    :param predicted: predicted data.
    :param ground_truth: Ground truth.
    :param coef: confidence coef.
    """
    correct = 0

    for i in range(np.shape(ground_truth)[0]):
        gt_sup = abs(ground_truth[i]) * (1 + coef)
        gt_inf = abs(ground_truth[i]) - abs(ground_truth[i]) * coef

        if abs(predicted[i]) > gt_inf and abs(predicted[i]) < gt_sup:
            correct += 1

    return correct

def mean_squared_error(preds, gt):
    """
    Calculates accuracy with a confidence coef.
    :param predicted: predicted data.
    :param ground_truth: Ground truth.
    """
    squared = 0.0
    total = 0
    aux = []
    for i in range(preds.shape[0]):
        squared += (preds[i] - gt[i])**2
        total += 1
        aux.append((preds[i] - gt[i])**2)

    return squared/total, max(aux), min(aux)
