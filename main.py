import argparse
from src.utils.image_utils import load_image, save_image
from src.dwt.dwt_embed import embed_dwt
from src.dwt.dwt_extract import extract_dwt
from src.dct.dct_embed import embed_dct
from src.dct.dct_extract import extract_dct
from src.utils.metrics import psnr


def main():
	parser = argparse.ArgumentParser(description="Embed and extract using DWT or DCT")
	parser.add_argument('--method', choices=['dwt', 'dct'], default='dwt', help='Steganography method')
	parser.add_argument('--input', default='data/input/images/input.png', help='Input image path')
	parser.add_argument('--output', default='data/output/stego.png', help='Output stego image path')
	args = parser.parse_args()

	image = load_image(args.input)
	secret = "Hello, this is a secret message hidden in the image using DCT steganography!"

	if args.method == 'dwt':
		stego = embed_dwt(image.copy(), secret)
		decoded = extract_dwt(stego)
	else:
		stego = embed_dct(image.copy(), secret)
		decoded = extract_dct(stego)

	save_image(args.output, stego)

	print("Extracted:", decoded)
	print("PSNR:", psnr(image, stego))


if __name__ == '__main__':
	main()

