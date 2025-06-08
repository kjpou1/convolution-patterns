# convolution_patterns/utils/argparse_utils.py

from typing import Sequence, Tuple
import argparse

def parse_image_size(values: Sequence[str | int]) -> Tuple[int, int]:
    """
    Validates and converts a list of two values into an (int, int) image size tuple.

    Args:
        values (Sequence): Typically a list like ['224', '224'] or [224, 224]

    Returns:
        Tuple[int, int]

    Raises:
        argparse.ArgumentTypeError
    """
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two integers for --image-size")
    try:
        return int(values[0]), int(values[1])
    except ValueError:
        raise argparse.ArgumentTypeError("Both values must be integers")
