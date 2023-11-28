from cv2 import COLOR_BGR2RGB, cvtColor, imread, merge, split
from numpy import float64, maximum
from numpy import max as np_max


def cargar_en_RGB(path: str):
    ret = imread(path)
    return cvtColor(ret, COLOR_BGR2RGB)


def tform_coord_cromaticas(img):
    # Separar canales de color y transformarlos a float
    red_channel, green_channel, blue_channel = split(img)

    red_channel = red_channel.astype(float64)
    green_channel = green_channel.astype(float64)
    blue_channel = blue_channel.astype(float64)

    # Aplicar formula f(RGB) = (R/(R+G+B), G/(R+G+B), B(R+G+B) )
    # Evitamos dividir entre 0 con un valor epsilon
    tot = maximum(red_channel + green_channel + blue_channel, 1e-10)

    red_channel = red_channel / tot
    green_channel = green_channel / tot
    blue_channel = blue_channel / tot

    # Fusionar los canales de nuevo en una imagen
    return merge([red_channel, green_channel, blue_channel])


def tform_white_patch(img):
    # Separar canales de color y transformarlos a float
    red_channel, green_channel, blue_channel = split(img)

    red_channel = red_channel.astype(float64)
    green_channel = green_channel.astype(float64)
    blue_channel = blue_channel.astype(float64)

    # Aplicar algoritmo white patch
    red_max = np_max(red_channel)
    green_max = np_max(green_channel)
    blue_max = np_max(blue_channel)

    red_channel = red_channel / red_max
    green_channel = green_channel / green_max
    blue_channel = blue_channel / blue_max

    # Fusionar los canales de nuevo en una imagen
    return merge([red_channel, green_channel, blue_channel])
