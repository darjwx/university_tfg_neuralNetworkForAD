#sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


#Pytorch
from torchvision.utils import make_grid

#Numpy
import numpy as np
import matplotlib.pyplot as plt

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/classifier_resnet')


def get_metrics(labels_true, labels_pred, num_classes, classes):

    recall = recall_score(labels_true, labels_pred, average=None, zero_division=0)
    precision = precision_score(labels_true, labels_pred, average=None, zero_division=0)
    f1 = f1_score(labels_true, labels_pred, average=None, zero_division=0)
    conf_matrix = confusion_matrix(labels_true, labels_pred)

    for i in range(num_classes):
        print('Recall, precision and F1-score of %5s: %1.3f %1.3f %1.3f'
              % (classes[i], recall[i], precision[i], f1[i]))

    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = classes)
    disp.plot()
    plt.show()


def show_predicted_data(dataloader, classes_1, classes_2, labels_pred_1, labels_pred_2):

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
    writer.add_scalar(tag, scalar, x)

def pr_curve_tb(num_classes, labels_1, labels_2, preds_1, preds_2):
    for i in range(num_classes):
        labels_i_1 = labels_1 == i
        labels_i_2 = labels_2 == i
        writer.add_pr_curve('Precision-Recall speed' + str(i), labels_i_1, preds_1[:,i], global_step=0)
        writer.add_pr_curve('Precision-Recall direction' + str(i), labels_i_2, preds_2[:,i], global_step=0)
