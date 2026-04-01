import pywt
import numpy as np
from src.utils.image_utils import to_binary

DELTA = 16
END_MARKER = '1111111111111110'


def dwt_capacity(image):
    if image.ndim == 3:
        channel = image[:, :, 0]
    else:
        channel = image
    coeffs = pywt.dwt2(channel, 'haar')
    LL, _ = coeffs
    return LL.shape[0] * LL.shape[1]


def embed_dwt_bits(channel, binary_secret, start_idx=0):
    """Embed bits into a single-channel (2D) image using DWT."""
    channel = np.float64(channel)
    coeffs = pywt.dwt2(channel, 'haar')
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
    """Embed secret into a color (BGR) image using DWT on the blue channel."""
    binary_secret = to_binary(secret) + END_MARKER

    # Work on the blue channel (index 0) to preserve color appearance
    channel = image[:, :, 0].copy()
    capacity = dwt_capacity(channel)

    if len(binary_secret) > capacity:
        raise ValueError(
            f"Message too large for image capacity. Bits needed: {len(binary_secret)}, capacity: {capacity}"
        )

    stego_channel, _ = embed_dwt_bits(channel, binary_secret)

    result = image.copy()
    result[:, :, 0] = stego_channel
    return result