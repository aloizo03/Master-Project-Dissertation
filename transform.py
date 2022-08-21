import cv2
import numpy as np


def transform(image, size, min_norm=0, max_norm=1., transpose=(2, 0, 1)):
    """
    Getting the image apply transform
    :param image: numpy, a numpy array of the image
    :param size: (int, int), the resize size of the image
    :param min_norm: float, the min value of normalization
    :param max_norm: flot, the max value of normalization
    :param transpose: (int, int, int), the transope dimansion of the image (form NxNx3 to 3xNxN)
    :return: numpy, the numpy array of the image transformation
    """
    img_resize = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    img_normalize = cv2.normalize(img_resize, None, min_norm, max_norm, cv2.NORM_MINMAX)
    img_normalize = img_normalize.transpose(transpose)
    return img_normalize
