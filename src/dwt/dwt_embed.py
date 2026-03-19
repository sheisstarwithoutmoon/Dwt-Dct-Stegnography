import pywt
import numpy as np
from src.utils.image_utils import to_binary

def embed_dwt(image, secret):
    binary_secret = to_binary(secret) + '1111111111111110'

    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    idx = 0
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            if idx < len(binary_secret):
                LL[i][j] = int(LL[i][j]) & ~1 | int(binary_secret[idx])
                idx += 1

    return pywt.idwt2((LL, (LH, HL, HH)), 'haar')