from pathlib import Path
import cv2
import numpy as np

COLS = 16, 316, 616, 916, 1216
ROWS = 18, 318, 618
HEIGHT = WEIGHT = 264

test_img_path = Path('~/Pictures/sesja_natka/sesja_Natka/DSC_8779.jpg').expanduser()
out_dir = Path("watermarks")

test_img = cv2.imread(str(test_img_path))

img_idx = 0
for row in ROWS:
    for col in COLS:
        roi = test_img[row:row+HEIGHT, col:col+WEIGHT]
        img_idx += 1
        roi_path = f"{test_img_path.name.replace('.jpg', '')}_{img_idx}.png"
        roi_full_path = out_dir / roi_path
        cv2.imwrite(str(roi_full_path), roi)

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        std = np.std(roi_gray)
        print(f"{roi_path} {std:.2f}")
