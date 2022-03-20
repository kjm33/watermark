from pathlib import Path
import cv2
import numpy as np

COLS = 16, 316, 616, 916, 1216
ROWS = 18, 318, 618
HEIGHT = WEIGHT = 264


def split_image(img_path: Path):
    img = cv2.imread(str(img_path))

    if img.shape != (1080, 1619, 3):
        return

    img_idx = 0
    for row in ROWS:
        for col in COLS:
            roi = img[row:row+HEIGHT, col:col+WEIGHT]
            img_idx += 1
            roi_path = f"{img_path.name.replace('.jpg', '')}_{img_idx}.png"
            roi_full_path = out_dir / roi_path
            cv2.imwrite(str(roi_full_path), roi)

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            std = np.std(roi_gray)
            # get top 10 chunks by
            # awk NF std_of_watermarks.txt | sort -g -k2,2 | head -n 10
            print(f"{roi_path} {std:.2f}")


out_dir = Path("watermarks")
input_dir = Path('~/Pictures/sesja_natka/sesja_Natka/').expanduser()

img_paths = input_dir.glob("*.jpg")

for img_path in sorted(img_paths):
    split_image(img_path)
    print()
