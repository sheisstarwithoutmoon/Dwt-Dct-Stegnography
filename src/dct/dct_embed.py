import cv2
import numpy as np
from src.utils.image_utils import to_binary

DELTA = 16

def embed_dct(image, secret):
    binary_secret = to_binary(secret) + '1111111111111110'

    image = np.float32(image)
    h, w = image.shape
    idx = 0

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = image[i:i+8, j:j+8]
            dct_block = cv2.dct(block)

            if idx < len(binary_secret):
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
        if idx >= len(binary_secret):
            break

    print(f"[DEBUG] Bits embedded: {idx} / {len(binary_secret)}")
    return np.clip(np.round(image), 0, 255).astype(np.uint8)