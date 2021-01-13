# OpenCV
import cv2 as cv

# Torch
import torch

class Rescale(object):
    """
    Class to Rescale images.
    """

    def __init__(self, output_size):
        """
        Loads the size of the rescaled image.
        :param output_size: Int tuple with the desired size.
        """

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Rescales the image.
        :param sample: Contains images/labels.
        """

        image, label = sample['image'], sample['label']

        h, w = self.output_size

        img = cv.resize(image, (h,w))

        return {'image': img, 'label': label}


class ToTensor(object):
    """
    Class to convert from numpy to tensor.
    """

    def __call__(self, sample):
        """
        Numpy to tensor.
        :param sample: Contains images/labels.
        """
        image, label = sample['image'], sample['label']

        # OpenCV image: H x W x C
        # torch image: C x H x W
        # The net expects: batch_number x C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
