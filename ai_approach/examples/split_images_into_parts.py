from pathlib import Path
import cv2
import numpy as np

SLICE_HEIGHT = SLICE_WIDTH = 264

out_dir = Path("cut_images")
input_dir = Path('./real_session/').expanduser()

img_paths = input_dir.glob("*.jpg")


def split_image(img_path: Path):
    img = cv2.imread(str(img_path))
    height, width, _ = img.shape

    img_idx = 0
    for row in range(0, height, SLICE_WIDTH):
        for col in range(0, width, SLICE_HEIGHT):
            abs_height = row + SLICE_HEIGHT
            abs_width = col + SLICE_WIDTH

            if abs_width > width or abs_height > height:
                continue

            roi = img[row:abs_height, col:abs_width]
            img_idx += 1
            roi_path = f"{img_path.name.replace('.jpg', '')}_{img_idx}.png"
            roi_full_path = out_dir / roi_path
            cv2.imwrite(str(roi_full_path), roi)


for img_path in sorted(img_paths):
    split_image(img_path)
    print()
