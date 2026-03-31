import cv2
import numpy as np
import pywt

from src.utils.image_utils import from_binary, to_binary

END_MARKER = '1111111111111110'
VIDEO_DELTA = 50  # Larger delta for robustness against video compression


def _embed_dwt_video_bits(image, binary_secret, start_idx=0):
    """Embed bits into a grayscale frame using DWT with a large delta."""
    image = np.float64(image)
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    idx = start_idx
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            if idx >= len(binary_secret):
                break

            coeff = LL[i][j]
            q = int(np.round(coeff / VIDEO_DELTA))
            bit = int(binary_secret[idx])

            if (q & 1) != bit:
                q += 1 if coeff >= 0 else -1

            LL[i][j] = float(q * VIDEO_DELTA)
            idx += 1

        if idx >= len(binary_secret):
            break

    stego = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    stego = np.clip(np.round(stego), 0, 255).astype(np.uint8)
    return stego, idx


def _extract_dwt_video_binary(image):
    """Extract binary data from a grayscale frame using DWT with a large delta."""
    image = np.float64(image)
    coeffs = pywt.dwt2(image, 'haar')
    LL, _ = coeffs

    binary_data = ""
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            coeff = LL[i][j]
            q = int(np.round(coeff / VIDEO_DELTA))
            binary_data += str(q & 1)

    return binary_data


def embed_dwt_video(input_video_path, output_video_path, secret):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found at path: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 24.0

    # Use mp4v codec which is universally supported by OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_video_path}")

    binary_secret = to_binary(secret) + END_MARKER
    idx = 0
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx < len(binary_secret):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stego_gray, idx = _embed_dwt_video_bits(gray, binary_secret, start_idx=idx)
            stego_frame = cv2.cvtColor(stego_gray, cv2.COLOR_GRAY2BGR)
            writer.write(stego_frame)
        else:
            writer.write(frame)

        frame_count += 1

    cap.release()
    writer.release()

    if idx < len(binary_secret):
        raise ValueError(
            f"Message too large for video capacity. Bits embedded: {idx}/{len(binary_secret)}"
        )

    return frame_count


def extract_dwt_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found at path: {video_path}")

    binary_data = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_data += _extract_dwt_video_binary(gray)

        end_idx = binary_data.find(END_MARKER)
        if end_idx != -1:
            cap.release()
            return from_binary(binary_data[:end_idx])

    cap.release()
    return ""
