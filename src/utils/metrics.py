import numpy as np

def mse(original, stego):
    return np.mean((original - stego) ** 2)

def psnr(original, stego):
    mse_val = mse(original, stego)
    if mse_val == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse_val))