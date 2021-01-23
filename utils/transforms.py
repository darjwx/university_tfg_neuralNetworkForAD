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

    def __init__(self, output_size, seq = False):
        """
        Loads the size of the rescaled image.
        :param output_size: Int tuple with the desired size.
        :param seq: True when working with a sequencial network.
        """

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seq = seq

    def __call__(self, sample):
        """
        Rescales the image.
        :param sample: Contains images/labels.
        """

        image, label = sample['image'], sample['label']

        h, w = self.output_size

        if self.seq:
            #print(np.shape(image))
            res_img = []
            for i in range(np.shape(image)[0]):
                res_img.append(cv.resize(image[i], (w,h)))

            #print(np.shape(res_img))
            return {'image': res_img, 'label': label}

        else:
            img = cv.resize(image, (w,h))

            return {'image': img, 'label': label}


class ToTensor(object):
    """
    Class to convert from numpy to tensor.
    """

    def __init__(self, seq = False):
        """
        Defines the network we are working with.
        :param seq: True when working with a sequencial network.
        """

        self.seq = seq

    def __call__(self, sample):
        """
        Numpy to tensor.
        :param sample: Contains images/labels.
        """
        image, label = sample['image'], sample['label']

        if self.seq:
            aux = []
            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            for i in range(np.shape(image)[0]):
                #print(np.shape(image))
                aux.append(image[i].transpose((2, 0, 1)))
                #print(np.shape(aux))

            return {'image': torch.from_numpy(np.array(aux)),
                    'label': torch.from_numpy(label)}
        else:
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

    def __init__(self, mean, std, seq = False):
        """
        Loads the mean and std.
        Defines if we are working with a sequencial network.
        :param mean: mean parameter.
        :param std: standar deviation parameter.
        :param seq: True when working with a sequencial network.
        """

        self.mean = mean
        self.std = std
        self.seq = self

    def __call__(self, sample):
        """
        Normalize each sample.
        :param sample: Contains images/labels.
        """
        image, label = sample['image'], sample['label']

        # Use Normalize from pytorch
        norm = transforms.Compose([transforms.Normalize(self.mean, self.std)])

        if self.seq:
            aux = torch.tensor([])
            for i in range(image.size(0)):
                aux = torch.cat((aux, norm(image[i])), dim=0)
        else:
            aux = norm(image)

        return {'image': aux,
                'label': label}
