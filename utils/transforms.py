"""
    Transforms - Classes to adapt input data. 
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

    def __init__(self, output_size, canbus = False, seq = False, reg = False, areg = False):
        """
        Loads the size of the rescaled image.
        :param output_size: Int tuple with the desired size.
        :param canbus: True when using CAN bus data as an input.
        :param seq: True when working with a sequencial network.
        :param reg: True when working with a regression model.
        :param areg: True when working with the aided regression model.
        """

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.canbus = canbus
        self.seq = seq
        self.reg = reg
        self.areg = areg

    def __call__(self, sample):
        """
        Rescales the image.
        :param sample: Contains images/labels, and optionally, canbus data.
        """

        h, w = self.output_size

        if self.seq:
            res_img = []
            image, label = sample['image'], sample['label']

            for i in range(np.shape(image)[0]):
                res_img.append(cv.resize(image[i], (w,h)))

            if self.canbus:
                return {'image': res_img, 'label': label, 'numerical': sample['numerical']}
            else:
                return {'image': res_img, 'label': label}

        elif self.reg:
            res_img = []
            image, canbus = sample['image'], sample['can_bus']

            for i in range(np.shape(image)[0]):
                res_img.append(cv.resize(image[i], (w,h)))

            return {'image': res_img, 'can_bus': canbus}

        elif self.areg:
            res_img = []
            image, sp_type, speed, st_type, steering = sample['image'], sample['sp_type'], sample['speed'], sample['st_type'], sample['steering']

            for i in range(np.shape(image)[0]):
                res_img.append(cv.resize(image[i], (w,h)))

            return {'image': res_img, 'sp_type': sp_type, 'speed': speed, 'st_type': st_type, 'steering': steering}

        else:
            image, label = sample['image'], sample['label']
            img = cv.resize(image, (w,h))

            if self.canbus:
                return {'image': img, 'label': label, 'numerical': sample['numerical']}
            else:
                return {'image': img, 'label': label}


class ToTensor(object):
    """
    Class to convert from numpy to tensor.
    """

    def __init__(self, canbus = False, seq = False, reg = False, areg = False):
        """
        Defines the network we are working with.
        :param canbus: True when using CAN bus data as an input.
        :param seq: True when working with a sequencial network.
        :param reg: True when working with a regression model.
        :param areg: True when working with the aided regression model.
        """

        self.canbus = canbus
        self.seq = seq
        self.reg = reg
        self.areg = areg

    def __call__(self, sample):
        """
        Numpy to tensor.
        :param sample: Contains images/labels, and optionally, canbus data.
        """

        if self.seq:
            aux = []
            image, label = sample['image'], sample['label']

            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            for i in range(np.shape(image)[0]):
                aux.append(image[i].transpose((2, 0, 1)))

            if self.canbus:
                return {'image': torch.from_numpy(np.array(aux)),
                        'label': torch.from_numpy(label),
                        'numerical': torch.from_numpy(np.array(sample['numerical']))}
            else:
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

        elif self.areg:
            aux = []
            image, sp_type, speed, st_type, steering = sample['image'], sample['sp_type'], sample['speed'], sample['st_type'], sample['steering']

            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            for i in range(np.shape(image)[0]):
                aux.append(image[i].transpose((2, 0, 1)))

            return {'image': torch.from_numpy(np.array(aux)),
                    'sp_type': torch.from_numpy(np.array(sp_type)),
                    'speed': torch.from_numpy(np.array(speed)).float(),
                    'st_type': torch.from_numpy(np.array(st_type)),
                    'steering': torch.from_numpy(np.array(steering)).float()}
        else:
            image, label = sample['image'], sample['label']

            # OpenCV image: H x W x C
            # torch image: C x H x W
            # The net expects: batch_number x C x H x W
            image = image.transpose((2, 0, 1))

            if self.canbus:
                return {'image': torch.from_numpy(image),
                        'label': torch.from_numpy(label),
                        'numerical': torch.from_numpy(sample['numerical'])}
            else:
                return {'image': torch.from_numpy(image),
                        'label': torch.from_numpy(label)}


class Normalize(object):
    """
    Class to normalize the mean and standard deviation.
    """

    def __init__(self, mean, std, mean_sp = 0, std_sp = 0, mean0 = 0, std0 = 0, mean1 = 0, std1 = 0, mean2 = 0, std2 = 0, canbus = False, seq = False, reg = False, areg = False):
        """
        Loads the mean and std.
        Defines if we are working with a sequencial network.
        :param mean: mean parameter.
        :param std: standard deviation parameter.
        :param mean1: Numerical mean parameter.
        :param std1: Numerical standard deviation parameter.
        :param mean2: Numerical mean parameter.
        :param std2: Numerical standard deviation parameter.
        :param canbus: True when using CAN bus data as an input.
        :param seq: True when working with a sequencial network.
        :param reg: True when working with a regression model.
        :param areg: True when working with the aided regression model.
        """

        self.mean = mean
        self.std = std
        self.std_sp = std_sp
        self.mean_sp = mean_sp
        self.std0 = std0
        self.mean0 = mean0
        self.std1 = std1
        self.mean1 = mean1
        self.std2 = std2
        self.mean2 = mean2
        self.canbus = canbus
        self.seq = seq
        self.reg = reg
        self.areg = areg

    def __call__(self, sample):
        """
        Normalize each sample.
        :param sample: Contains images/labels, and optionally, canbus data.
        """

        # Use Normalize from pytorch
        norm = transforms.Compose([transforms.Normalize(self.mean, self.std)])

        if self.seq:
            image, label = sample['image'], sample['label']

            for i in range(image.size(0)):
                image[i] = norm(image[i])

            if self.canbus:
                return {'image': image,
                        'label': label,
                        'numerical': sample['numerical']}
            else:
                return {'image': image,
                        'label': label}

        elif self.reg:
            image, canbus = sample['image'], sample['can_bus']
            for i in range(image.size(0)):
                image[i] = norm(image[i])

            # Normalize numerical values
            for i in range(np.shape(canbus)[0]):

                canbus[i,0] = (canbus[i,0] - self.mean_sp) / self.std_sp
                canbus[i,1] = (canbus[i,1] - self.mean0) / self.std0

            return {'image': image,
                    'can_bus': canbus}

        elif self.areg:
            image, sp_type, speed, st_type, steering = sample['image'], sample['sp_type'], sample['speed'], sample['st_type'], sample['steering']
            for i in range(image.size(0)):
                image[i] = norm(image[i])

            # Normalize per speed class
            for i in range(np.shape(speed)[0]):
                if sp_type[i] == 1:
                    speed[i] = (speed[i] - self.mean_sp) / self.std_sp


            # Normalize per steering class
            for i in range(np.shape(steering)[0]):
                if st_type[i] == 0:
                    steering[i] = (steering[i] - self.mean0) / self.std0
                elif st_type[i] == 1:
                    steering[i] = (steering[i] - self.mean1) / self.std1
                else:
                    steering[i] = (steering[i] - self.mean2) / self.std2

            return {'image': image, 'sp_type': sp_type, 'speed': speed, 'st_type': st_type, 'steering': steering}

        else:
            image, label = sample['image'], sample['label']

            image = norm(image)

            if self.canbus:
                return {'image': image,
                        'label': label,
                        'numerical': sample['numerical']}
            else:
                return {'image': image,
                        'label': label}
