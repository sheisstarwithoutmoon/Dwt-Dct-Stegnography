from src.utils.image_utils import load_image, save_image
from src.dwt.dwt_embed import embed_dwt
from src.dwt.dwt_extract import extract_dwt
from src.dct.dct_embed import embed_dct
from src.dct.dct_extract import extract_dct
from src.dct.dct_video import embed_dct_video, extract_dct_video
from src.dwt.dwt_video import embed_dwt_video, extract_dwt_video
from src.utils.metrics import psnr, video_psnr


def _run_image(method):
	default_input = 'data/input/images/input.png'
	default_output = f'data/output/stego_{method}.png'
	mode = input("Mode - embed or extract [e/x]: ").strip().lower()

	if mode not in ('e', 'x'):
		raise ValueError("Invalid mode. Enter 'e' for embed or 'x' for extract.")

	input_path = input(f"Input image path [{default_input}]: ").strip() or default_input
	image = load_image(input_path)

	if mode == 'x':
		if method == 'dct':
			decoded = extract_dct(image)
		else:
			decoded = extract_dwt(image)

		if decoded:
			print("Extracted:", decoded)
		else:
			print("No hidden message found (or wrong method selected).")
		return

	output_path = input(f"Output stego image path [{default_output}]: ").strip() or default_output
	secret = input("Secret message: ").strip()

	if not secret:
		raise ValueError("Secret message cannot be empty.")

	if method == 'dct':
		stego = embed_dct(image.copy(), secret)
		decoded = extract_dct(stego)
	else:
		stego = embed_dwt(image.copy(), secret)
		decoded = extract_dwt(stego)

	save_image(output_path, stego)

	print("Output image:", output_path)
	print("Extracted:", decoded)
	print("PSNR:", psnr(image, stego))


def _run_video(method):
	default_input = 'data/input/videos/input.mp4'
	default_output = f'data/output/stego_{method}.mp4'
	mode = input("Mode - embed or extract [e/x]: ").strip().lower()

	if mode not in ('e', 'x'):
		raise ValueError("Invalid mode. Enter 'e' for embed or 'x' for extract.")

	input_path = input(f"Input video path [{default_input}]: ").strip() or default_input

	if mode == 'x':
		if method == 'dct':
			decoded = extract_dct_video(input_path)
		else:
			decoded = extract_dwt_video(input_path)

		if decoded:
			print("Extracted:", decoded)
		else:
			print("No hidden message found (or wrong method selected).")
		return

	output_path = input(f"Output stego video path [{default_output}]: ").strip() or default_output
	secret = input("Secret message: ").strip()

	if not secret:
		raise ValueError("Secret message cannot be empty.")

	if method == 'dct':
		embed_dct_video(input_path, output_path, secret)
		decoded = extract_dct_video(output_path)
	else:
		embed_dwt_video(input_path, output_path, secret)
		decoded = extract_dwt_video(output_path)

	print("Output video:", output_path)
	print("Extracted:", decoded)
	# print("PSNR:", video_psnr(input_path, output_path))


def main():
	print("Select operation:")
	print("1. DCT on image")
	print("2. DWT on image")
	print("3. DCT on video")
	print("4. DWT on video")

	choice = input("Enter 1/2/3/4: ").strip()

	if choice == '1':
		_run_image('dct')
	elif choice == '2':
		_run_image('dwt')
	elif choice == '3':
		_run_video('dct')
	elif choice == '4':
		_run_video('dwt')
	else:
		print("Invalid choice. Please run again and enter 1, 2, 3, or 4.")


if __name__ == '__main__':
	try:
		main()
	except Exception as exc:
		print("Error:", exc)

