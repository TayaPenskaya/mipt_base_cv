import imageio
import numpy as np
from PIL import Image


def get_image_from_bmp(img_path):
    return np.array(imageio.imread(img_path), dtype=float)


def get_image_from_array(arr):
    return Image.fromarray(arr)
