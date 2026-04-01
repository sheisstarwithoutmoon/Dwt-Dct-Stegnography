import cv2
import numpy as np
from src.utils.image_utils import to_binary

DELTA = 16
END_MARKER = '1111111111111110'


def dct_capacity(image):
    if image.ndim == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape
    return ((h - 7) // 8) * ((w - 7) // 8)


def embed_dct_bits(channel, binary_secret, start_idx=0):
    """Embed bits into a single-channel (2D) grayscale image using DCT."""
    channel = np.float32(channel)
    h, w = channel.shape
    idx = start_idx

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            if idx >= len(binary_secret):
                break

            block = channel[i:i+8, j:j+8]
            dct_block = cv2.dct(block)

            coeff = dct_block[4][5]
            q = int(np.round(coeff / DELTA))
            bit = int(binary_secret[idx])

            if (q & 1) != bit:
                q += 1 if coeff >= 0 else -1

            dct_block[4][5] = float(q * DELTA)
            idx += 1

            channel[i:i+8, j:j+8] = cv2.idct(dct_block)

        if idx >= len(binary_secret):
            break

    stego = np.clip(np.round(channel), 0, 255).astype(np.uint8)
    return stego, idx


def embed_dct(image, secret):
    """Embed secret into a color (BGR) image using DCT on the blue channel."""
    binary_secret = to_binary(secret) + END_MARKER

    # Work on the blue channel (index 0) to preserve color appearance
    channel = image[:, :, 0].copy()
    capacity = dct_capacity(channel)

    if len(binary_secret) > capacity:
        raise ValueError(
            f"Message too large for image capacity. Bits needed: {len(binary_secret)}, capacity: {capacity}"
        )

    stego_channel, _ = embed_dct_bits(channel, binary_secret)

    result = image.copy()
    result[:, :, 0] = stego_channel
    return result