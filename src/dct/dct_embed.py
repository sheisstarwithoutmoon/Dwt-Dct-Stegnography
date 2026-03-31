import cv2
import numpy as np
from src.utils.image_utils import to_binary

DELTA = 16
END_MARKER = '1111111111111110'


def dct_capacity(image):
    h, w = image.shape
    return ((h - 7) // 8) * ((w - 7) // 8)


def embed_dct_bits(image, binary_secret, start_idx=0):
    image = np.float32(image)
    h, w = image.shape
    idx = start_idx

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            if idx >= len(binary_secret):
                break

            block = image[i:i+8, j:j+8]
            dct_block = cv2.dct(block)

            coeff = dct_block[4][5]
            q = int(np.round(coeff / DELTA))
            bit = int(binary_secret[idx])

            if (q & 1) != bit:
                q += 1 if coeff >= 0 else -1

            dct_block[4][5] = float(q * DELTA)
            idx += 1

            image[i:i+8, j:j+8] = cv2.idct(dct_block)

        if idx >= len(binary_secret):
            break

    stego = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return stego, idx

def embed_dct(image, secret):
    binary_secret = to_binary(secret) + END_MARKER
    capacity = dct_capacity(image)

    if len(binary_secret) > capacity:
        raise ValueError(
            f"Message too large for image capacity. Bits needed: {len(binary_secret)}, capacity: {capacity}"
        )

    stego, idx = embed_dct_bits(image, binary_secret)
    return stego