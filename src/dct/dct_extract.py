import cv2
import numpy as np
from src.utils.image_utils import from_binary

DELTA = 16
END_MARKER = '1111111111111110'


def extract_dct_binary(channel):
    """Extract binary data from a single-channel (2D) image using DCT."""
    binary_data = ""
    h, w = channel.shape

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = np.float32(channel[i:i+8, j:j+8])
            dct_block = cv2.dct(block)

            coeff = dct_block[4][5]
            q = int(np.round(coeff / DELTA))
            binary_data += str(q & 1)

    return binary_data


def extract_dct(image):
    """Extract hidden message from a color (BGR) image using DCT on the blue channel."""
    channel = image[:, :, 0]
    binary_data = extract_dct_binary(channel)
    end_idx = binary_data.find(END_MARKER)
    if end_idx != -1:
        return from_binary(binary_data[:end_idx])

    return ""