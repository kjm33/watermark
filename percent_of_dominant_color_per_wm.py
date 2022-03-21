from pathlib import Path
import cv2
import numpy as np
from collections import Counter

input_dir = Path("./watermarks")
wm_paths = input_dir.glob("*png")


for wm_path in wm_paths:
    wm = cv2.imread(str(wm_path))
    unique, counts = np.unique(wm.reshape(-1, 3), axis=0, return_counts=True)
    idx = np.argmax(counts)
    freq = counts[idx]
    value = unique[idx]
    percent = freq / (wm.shape[0]*wm.shape[1])*100
    print(f"{wm_path} {percent:.2f} {value}")
