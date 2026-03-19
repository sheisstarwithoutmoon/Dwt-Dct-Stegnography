import pywt
from src.utils.image_utils import from_binary

def extract_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, _ = coeffs

    binary_data = ""

    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            binary_data += str(int(LL[i][j]) & 1)

            if binary_data.endswith('1111111111111110'):
                return from_binary(binary_data[:-16])

    return ""