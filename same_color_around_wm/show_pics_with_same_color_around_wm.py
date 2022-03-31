from pathlib import Path
import cv2
import numpy as np
from collections import Counter

input_dir = Path("../watermarks")
wm_paths = input_dir.glob("*png")

# roi = img[row:row+HEIGHT, col:col+WEIGHT]
# top_roi = [39:40, 86:87]

for wm_path in wm_paths:
    wm_bgr = cv2.imread(str(wm_path))
    wm = cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2RGB)

    top_roi = wm[30:35, 86:87]
    bottom_roi = wm[55:60, 86:87]
    if not np.array_equal(top_roi, bottom_roi):
        continue

    unique_pixels = set([tuple(e[0]) for e in top_roi.tolist()])

    if len(unique_pixels) != 1:  # all pixels aren't the same
        continue

    modified = wm[36:54, 86:87]

    print(f"{wm_path} color around {unique_pixels}")
