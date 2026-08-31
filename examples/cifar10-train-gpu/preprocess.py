import numpy as np


def preprocess(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]
