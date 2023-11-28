from cv2 import (
    COLOR_BGR2HSV,
    COLOR_BGR2RGB,
    IMREAD_GRAYSCALE,
    cvtColor,
    imread,
    merge,
    split,
)
from numpy import float64, max, maximum


def load_as_RGB(path: str):
    ret = imread(path)
    return cvtColor(ret, COLOR_BGR2RGB)


def load_as_gray(path: str):
    ret = imread(path, IMREAD_GRAYSCALE)
    return ret


def load_as_HSV(path: str):
    ret = imread(path)
    return cvtColor(ret, COLOR_BGR2HSV)


def tform_crom_coord(img):
    # Split image in RGB channels and set them to float to prepare them for arithmetic operations
    red_channel, green_channel, blue_channel = split(img)

    red_channel = red_channel.astype(float64)
    green_channel = green_channel.astype(float64)
    blue_channel = blue_channel.astype(float64)

    # Apply formula f(RGB) = (R/(R+G+B), G/(R+G+B), B(R+G+B) )
    # Avoid zero division using epsilon
    tot = maximum(red_channel + green_channel + blue_channel, 1e-10)

    red_channel = red_channel / tot
    green_channel = green_channel / tot
    blue_channel = blue_channel / tot

    # Merge channels to image
    return merge([red_channel, green_channel, blue_channel])


def tform_white_patch(img):
    # Split image in RGB channels and set them to float to prepare them for arithmetic operations
    red_channel, green_channel, blue_channel = split(img)

    red_channel = red_channel.astype(float64)
    green_channel = green_channel.astype(float64)
    blue_channel = blue_channel.astype(float64)

    # Apply white patch algorithm
    red_max = max(red_channel)
    green_max = max(green_channel)
    blue_max = max(blue_channel)

    red_channel = red_channel / red_max
    green_channel = green_channel / green_max
    blue_channel = blue_channel / blue_max

    # Merge channels to image
    return merge([red_channel, green_channel, blue_channel])
