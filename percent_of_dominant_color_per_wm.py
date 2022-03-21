from pathlib import Path
import cv2
import numpy as np
from collections import Counter

input_dir = Path("./watermarks")
wm_paths = input_dir.glob("*.png")


for wm_path in wm_paths:
    wm_gray = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
    unique, counts = np.unique(wm_gray.flatten(), return_counts=True)
    freq = np.argmax(counts)
    value = unique[freq]
    percent = freq / (wm_gray.shape[0]*wm_gray.shape[1])*100
    print(f"{wm_path} {percent:.2f} {value}")
