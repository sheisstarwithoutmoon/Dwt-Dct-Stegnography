import cv2
import numpy as np


def mse(original, stego):
    return np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2)

def psnr(original, stego):
    mse_val = mse(original, stego)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

def video_psnr(original_path, stego_path):
    """Compute average PSNR between two videos by comparing frames."""
    cap_orig = cv2.VideoCapture(original_path)
    cap_stego = cv2.VideoCapture(stego_path)

    if not cap_orig.isOpened():
        raise FileNotFoundError(f"Original video not found: {original_path}")
    if not cap_stego.isOpened():
        raise FileNotFoundError(f"Stego video not found: {stego_path}")

    psnr_values = []

    while True:
        ok1, frame_orig = cap_orig.read()
        ok2, frame_stego = cap_stego.read()

        if not ok1 or not ok2:
            break

        frame_psnr = psnr(frame_orig, frame_stego)
        psnr_values.append(frame_psnr)

    cap_orig.release()
    cap_stego.release()

    if not psnr_values:
        return 0.0

    return np.mean(psnr_values)