import cv2
import numpy as np
from src.utils.image_utils import from_binary

DELTA = 16

def extract_dct(image):
    binary_data = ""
    h, w = image.shape

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = np.float32(image[i:i+8, j:j+8])
            dct_block = cv2.dct(block)

            coeff = dct_block[4][5]
            q = int(np.round(coeff / DELTA))
            binary_data += str(q & 1)

            if len(binary_data) >= 16 and binary_data[-16:] == '1111111111111110':
                return from_binary(binary_data[:-16])

    return ""