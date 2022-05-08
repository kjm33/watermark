import sys
import argparse
from pathlib import Path
import cv2
import numpy as np


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--column", type=int, required=True,
                        help="NUmber of column to dump")
    return parser.parse_args()


args = get_params()
COLUMN = args.column

input_dir = Path("./examples")
examples_paths = input_dir.glob("*.png")

header = [" "] + list(map(str, range(1, 264)))
print(", ".join(header))

for img_path in examples_paths:
    img_gray = cv2.imread(str(img_path), 0)
    roi = img_gray[0:264, COLUMN:COLUMN+1]
    roi_str = [str(i[0]) for i in roi]
    row = [img_path.name] + roi_str
    print(", ".join(row))
