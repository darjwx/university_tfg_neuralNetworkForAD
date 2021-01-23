# sklearn metrics
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Pytorch
from torchvision.utils import make_grid

# Numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
# Change the route when testing different models
writer = SummaryWriter('runs/')


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


def show_predicted_data(dataloader, classes_1, classes_2, labels_pred_1, labels_pred_2):
    """
    Shows a batch of images with their corresponding prediction and ground truth.
    :param dataloader: Dataloader variable from pytorch with images and labels.
    :param classes_1: Number of classes 1.
    :param classes_2: Number of classes 2.
    :param labels_pred_1: Predictions 1.
    :param labels_pred_2: Predictions 2.
    """

    # Adapts the image for plt.imshow
    def imshow(image):
        img = image
        # Normalize values
        img = img.numpy()
        img = img / 255
        # imshow expects and array with the form: H * W * Colour channels
        img = np.transpose(img, (1, 2, 0))
        # BGR -> RGB
        plt.imshow(img[...,::-1])
        plt.show()

    labels_pred_1 = labels_pred_1.numpy().astype(np.uint8)
    labels_pred_2 = labels_pred_2.numpy().astype(np.uint8)

    for i, data in enumerate(dataloader):
        images = data['image']
        labels = data['label']

        labels = labels.numpy()

        print('------')
        for j in range(np.shape(labels)[0]):
            info = 'predicted: {} gt: {}'.format(classes_1[labels_pred_1[j]], classes_1[labels[j,0]])
            print(str(j), 'Speed: ', info)

            info = 'predicted: {} gt: {}'.format(classes_2[labels_pred_2[j]], classes_2[labels[j,1]])
            print('  Steering: ', info)
        print('------')

        imshow(make_grid(images))

def update_scalar_tb(tag, scalar, x):
    """
    Writes a scalar value to TensorBoard
    :param tag: Name of the graph.
    :param scalar: Scalar to write.
    :param x: X axis.
    """

    writer.add_scalar(tag, scalar, x)

def pr_curve_tb(num_classes, labels_1, labels_2, preds_1, preds_2):
    """
    Writes a per class precision-recall curve to TensorBoard.
    :param num_classes: Number of classes.
    :param labels_1: List of labels 1.
    :param labels_2: List of labels 2.
    :param preds_1: List of predictions 1.
    :param preds_2: List of predictions 2.
    """

    for i in range(num_classes):
        labels_i_1 = labels_1 == i
        labels_i_2 = labels_2 == i

        writer.add_pr_curve('Precision-Recall speed' + str(i), labels_i_1, preds_1[:,i], global_step=0)
        writer.add_pr_curve('Precision-Recall direction' + str(i), labels_i_2, preds_2[:,i], global_step=0)

def visualize_model(net, images):
    """
    Visualise the model in TensorBoard.
    :param net: Neural network model.
    :param images: Input of the model.
    """

    writer.add_graph(net, images)
    writer.close()
