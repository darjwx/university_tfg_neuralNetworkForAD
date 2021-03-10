# OpenCV
import cv2 as cv

# Torch
import torch

# Numpy
import numpy as np

import torchvision.transforms as transforms

class Rescale(object):
    """
    Class to Rescale images.
    """

    def __init__(self, output_size, seq = False, reg = False):
        """
        Loads the size of the rescaled image.
        :param output_size: Int tuple with the desired size.
        :param seq: True when working with a sequencial network.
        :param reg: True when working with a regression model.
        """

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seq = seq
        self.reg = reg

    def __call__(self, sample):
        """
        Rescales the image.
        :param sample: Contains images/labels.
        """

        h, w = self.output_size

        if self.seq:
            res_img = []
            image, label = sample['image'], sample['label']

            for i in range(np.shape(image)[0]):
                res_img.append(cv.resize(image[i], (w,h)))

            return {'image': res_img, 'label': label}
        elif self.reg:
            res_img = []
            image, canbus = sample['image'], sample['can_bus']

            for i in range(np.shape(image)[0]):
                res_img.append(cv.resize(image[i], (w,h)))

            return {'image': res_img, 'can_bus': canbus}

        else:
            image, label = sample['image'], sample['label']
            img = cv.resize(image, (w,h))

            return {'image': img, 'label': label}


class ToTensor(object):
    """
    Class to convert from numpy to tensor.
    """

    def __init__(self, seq = False, reg = False):
        """
        Defines the network we are working with.
        :param seq: True when working with a sequencial network.
        :param reg: True when working with a regression model.
        """

        self.seq = seq
        self.reg = reg

    def __call__(self, sample):
        """
        Numpy to tensor.
        :param sample: Contains images/labels.
        """

        if self.seq:
            aux = []
            image, label = sample['image'], sample['label']

            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            for i in range(np.shape(image)[0]):
                aux.append(image[i].transpose((2, 0, 1)))

            return {'image': torch.from_numpy(np.array(aux)),
                    'label': torch.from_numpy(label)}
        elif self.reg:
            aux = []
            image, canbus = sample['image'], sample['can_bus']

            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            for i in range(np.shape(image)[0]):
                aux.append(image[i].transpose((2, 0, 1)))

            return {'image': torch.from_numpy(np.array(aux)),
                    'can_bus': torch.from_numpy(np.array(canbus))}
        else:
            image, label = sample['image'], sample['label']

            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            image = image.transpose((2, 0, 1))

            return {'image': torch.from_numpy(image),
                    'label': torch.from_numpy(label)}


class Normalize(object):
    """
    Class to normalize the mean and standard deviation.
    """

    def __init__(self, mean, std, mean1 = 0, std1 = 0, mean2 = 0, std2 = 0, seq = False, reg = False):
        """
        Loads the mean and std.
        Defines if we are working with a sequencial network.
        :param mean: mean parameter.
        :param std: standard deviation parameter.
        :param mean1: Numerical mean parameter.
        :param std1: Numerical standard deviation parameter.
        :param mean2: Numerical mean parameter.
        :param std2: Numerical standar deviation parameter.
        :param seq: True when working with a sequencial network.
        :param reg: True when working with a regression model.
        """

        self.mean = mean
        self.std1 = std1
        self.mean1 = mean1
        self.std2 = std2
        self.mean2 = mean2
        self.std = std
        self.seq = seq
        self.reg = reg

    def __call__(self, sample):
        """
        Normalize each sample.
        :param sample: Contains images/labels.
        """

        # Use Normalize from pytorch
        norm = transforms.Compose([transforms.Normalize(self.mean, self.std)])

        if self.seq:
            image, label = sample['image'], sample['label']

            for i in range(image.size(0)):
                image[i] = norm(image[i])

            return {'image': image,
                    'label': label}
        elif self.reg:
            image, canbus = sample['image'], sample['can_bus']
            for i in range(image.size(0)):
                image[i] = norm(image[i])

            # Normalize numerical values
            for i in range(np.shape(canbus)[0]):

                canbus[i,0] = (canbus[i,0] - self.mean1) / self.std1
                canbus[i,1] = (canbus[i,1] - self.mean2) / self.std2

            return {'image': image,
                    'can_bus': canbus}
        else:
            image, label = sample['image'], sample['label']

            img = norm(image)

            return {'image': img,
                    'label': label}
