import pywt
import numpy as np
from src.utils.image_utils import to_binary

DELTA = 16
END_MARKER = '1111111111111110'


def dwt_capacity(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, _ = coeffs
    return LL.shape[0] * LL.shape[1]


def embed_dwt_bits(image, binary_secret, start_idx=0):
    image = np.float64(image)
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    idx = start_idx
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            if idx >= len(binary_secret):
                break

            coeff = LL[i][j]
            q = int(np.round(coeff / DELTA))
            bit = int(binary_secret[idx])

            if (q & 1) != bit:
                q += 1 if coeff >= 0 else -1

            LL[i][j] = float(q * DELTA)
            idx += 1

        if idx >= len(binary_secret):
            break

    stego = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    stego = np.clip(np.round(stego), 0, 255).astype(np.uint8)
    return stego, idx


def embed_dwt(image, secret):
    binary_secret = to_binary(secret) + END_MARKER
    capacity = dwt_capacity(image)

    if len(binary_secret) > capacity:
        raise ValueError(
            f"Message too large for image capacity. Bits needed: {len(binary_secret)}, capacity: {capacity}"
        )

    stego, _ = embed_dwt_bits(image, binary_secret)
    return stego