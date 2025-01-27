from __future__ import division

import logging
import numbers
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from torchvision.transforms import functional as F
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
)

logger = logging.getLogger("global")



# Gamma Transform
class Gamma(object):
    def __init__(self, gamma_range, prob):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            gamma_table = [np.power(x / 255.0, 1 / gamma) * 255 for x in range(256)]
            gamma_table = np.array(gamma_table).astype(np.uint8)
            img = cv2.LUT(img, gamma_table).astype(np.uint8)

        return img


# Gray Scale
class RandomGrayscale(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img):
        flag = torch.rand(1)[0].item() < self.prob
        if flag:
            return F.to_grayscale(img, num_output_channels=3)
        else:
            return img


# Horizontal Flip
class RandomHFlip(object):
    def __init__(self, img_size, prob):
        self.img_size = img_size
        self.prob = prob

    def __call__(self, img,  boxes, points):
        flip_flag = torch.rand(1)[0].item() < self.prob
        if flip_flag:
            h, w = self.img_size
            # flip boxes
            boxes_temp = []
            for box_rsz in boxes:
                x_tl_, y_tl_ = box_rsz[0]
                x_br_, y_br_ = box_rsz[1]
                x_tl = w - x_br_
                x_br = w - x_tl_
                box_rsz[0] = x_tl, y_tl_
                box_rsz[1] = x_br, y_br_
                boxes_temp.append(box_rsz)
            boxes = boxes_temp
            # flip points
            points_temp = []
            for point in points:
                x, y = point
                x = w - x
                points_temp.append([x, y])
            points = points_temp
            return F.hflip(img),  boxes, points
        else:
            return img, boxes, points


# Vertical Flip
class RandomVFlip(object):
    def __init__(self, img_size, prob):
        self.img_size = img_size
        self.prob = prob

    def __call__(self, img, boxes, points):
        flip_flag = torch.rand(1)[0].item() < self.prob
        if flip_flag:
            h, w = self.img_size
            # flip boxes
            boxes_temp = []
            for box_rsz in boxes:
                x_tl_, y_tl_ = box_rsz[0]
                x_br_, y_br_ = box_rsz[1]
                y_tl = h - y_br_
                y_br = h - y_tl_
                box_rsz[0] = x_tl_, y_tl
                box_rsz[1] = x_br_, y_br
                boxes_temp.append(box_rsz)
            boxes = boxes_temp
            # flip points
            points_temp = []
            for point in points:
                x, y = point
                y = h - y
                points_temp.append([x, y])
            points = points_temp
            return F.vflip(img),  boxes, points
        else:
            return img, boxes, points


# Random Color Jittering
class RandomColorJitter(object):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Arguments:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.prob = prob

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self, brightness, contrast, saturation, hue):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        img_transforms = []

        if brightness is not None and random.random() < self.prob:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )

        if contrast is not None and random.random() < self.prob:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor))
            )

        if saturation is not None and random.random() < self.prob:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )

        if hue is not None and random.random() < self.prob:
            hue_factor = random.uniform(hue[0], hue[1])
            img_transforms.append(
                transforms.Lambda(lambda img: adjust_hue(img, hue_factor))
            )

        random.shuffle(img_transforms)
        img_transforms = transforms.Compose(img_transforms)

        return img_transforms

    def __call__(self, img):
        """
        Arguments:
            img (PIL Image): Input image.
        Returns:
            img (PIL Image): Color jittered image.
        """
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img = transform(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        return format_string

    @classmethod
    def from_params(cls, brightness, contrast, saturation, hue, prob):
        brightness = brightness
        contrast = contrast
        hue = hue
        saturation = saturation
        prob = prob
        return cls(
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation,
            prob=prob,
        )

