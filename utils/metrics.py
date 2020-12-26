#sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


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
