import argparse

from src.dct.dct_extract import extract_dct
from src.dwt.dwt_extract import extract_dwt
from src.utils.image_utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden text from a stego image using DCT or DWT."
    )
    parser.add_argument(
        "image_path",
        help="Path to the image that may contain a hidden message.",
    )
    parser.add_argument(
        "--method",
        choices=["dct", "dwt"],
        default="dct",
        help="Steganography method used to hide the message (default: dct).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.image_path)

    if args.method == "dct":
        message = extract_dct(image)
    else:
        message = extract_dwt(image)

    if message:
        print(f"Method: {args.method}")
        print(f"Extracted message: {message}")
    else:
        print(
            "No message found (or wrong method was selected). "
            "Try the other method if unsure."
        )


if __name__ == "__main__":
    main()
