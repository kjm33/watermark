from pathlib import Path
import cv2
import numpy as np

COLS = 16, 316, 616, 916, 1216
ROWS = 18, 318, 618
HEIGHT = WEIGHT = 264

test_img_path = Path('~/Pictures/sesja_natka/sesja_Natka/DSC_8779.jpg').expanduser()
out = Path("watermarks")

test_img = cv2.imread(str(test_img_path))

for row in ROWS:
    for col in COLS:
        roi = test_img[row:row+HEIGHT, col:col+WEIGHT]
        roi_path = f"{row}_{col}.png"
        roi_full_path = out / roi_path
        cv2.imwrite(str(roi_full_path), roi)