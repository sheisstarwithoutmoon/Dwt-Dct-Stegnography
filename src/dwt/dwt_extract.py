import pywt
import numpy as np
from src.utils.image_utils import from_binary

DELTA = 16
END_MARKER = '1111111111111110'


def extract_dwt_binary(channel):
    """Extract binary data from a single-channel (2D) image using DWT."""
    channel = np.float64(channel)
    coeffs = pywt.dwt2(channel, 'haar')
    LL, _ = coeffs

    binary_data = ""
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            coeff = LL[i][j]
            q = int(np.round(coeff / DELTA))
            binary_data += str(q & 1)

    return binary_data


def extract_dwt(image):
    """Extract hidden message from a color (BGR) image using DWT on the blue channel."""
    channel = image[:, :, 0]
    binary_data = extract_dwt_binary(channel)
    end_idx = binary_data.find(END_MARKER)
    if end_idx != -1:
        return from_binary(binary_data[:end_idx])

    return ""