#sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

#Pytorch
from torchvision.utils import make_grid

#Numpy
import numpy as np
import matplotlib.pyplot as plt


def get_metrics(labels_true, labels_pred, num_classes, classes):

    recall = recall_score(labels_true, labels_pred, average=None, zero_division=0)
    precision = precision_score(labels_true, labels_pred, average=None, zero_division=0)
    f1 = f1_score(labels_true, labels_pred, average=None, zero_division=0)
    conf_matrix = confusion_matrix(labels_true, labels_pred)

    for i in range(num_classes):
        print('Recall, precision and F1-score of %5s: %1.3f %1.3f %1.3f'
              % (classes[i], recall[i], precision[i], f1[i]))


    print('--Confusion matrix--')
    print(conf_matrix)

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
