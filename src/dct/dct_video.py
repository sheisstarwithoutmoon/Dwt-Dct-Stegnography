import cv2

from src.dct.dct_embed import END_MARKER, embed_dct_bits
from src.dct.dct_extract import extract_dct_binary
from src.utils.image_utils import from_binary, to_binary


def embed_dct_video(input_video_path, output_video_path, secret):
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
            stego_gray, idx = embed_dct_bits(gray, binary_secret, start_idx=idx)
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


def extract_dct_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found at path: {video_path}")

    binary_data = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary_data += extract_dct_binary(gray)

        end_idx = binary_data.find(END_MARKER)
        if end_idx != -1:
            cap.release()
            return from_binary(binary_data[:end_idx])

    cap.release()
    return ""
