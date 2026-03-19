from src.utils.image_utils import load_image, save_image
from src.dct.dct_embed import embed_dct
from src.dct.dct_extract import extract_dct
from src.utils.metrics import psnr

image = load_image("data/input/images/input.png")

secret = "Hello Vanya"

stego = embed_dct(image.copy(), secret)

save_image("data/output/stego.png", stego)

decoded = extract_dct(stego)
print("Extracted:", decoded)

print("PSNR:", psnr(image, stego))

