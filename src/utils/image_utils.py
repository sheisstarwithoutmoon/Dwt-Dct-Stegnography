import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return img

def save_image(path, image):
    cv2.imwrite(path, image)

def to_binary(data):
    result = ""
    for char in data:
        result += format(ord(char), '08b')
    return result

def from_binary(binary_data):
    # make sure length is multiple of 8
    n = (len(binary_data) // 8) * 8
    binary_data = binary_data[:n]
    
    result = ""
    for i in range(0, n, 8):
        byte = binary_data[i:i+8]
        result += chr(int(byte, 2))
    return result