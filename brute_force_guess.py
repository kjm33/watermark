from pathlib import Path
import cv2
import numpy as np
from decimal import Decimal

one = Decimal('1.0')

ALPHA = 0.85
BETA = 0.15

files = ["./watermarks/DSC_8697_4.png", "./watermarks/DSC_8779_10.png"]
background_position = (111, 19)
wm_pos = (42, 21)

f1 = cv2.imread(files[0])
b1 = f1[background_position][0]
wm1 = f1[wm_pos][0]
# print(b1)
# print(wm1)

f2 = cv2.imread(files[1])
b2 = f2[background_position][0]
wm2 = f2[wm_pos][0]


for wm_pixel in range(256):
    for gamma in range(256):
        dst1 = cv2.addWeighted(src1=np.array(b1, dtype=float), alpha=float(ALPHA), src2=np.array(wm_pixel, dtype=float), beta=float(BETA), gamma=gamma)
        dst2 = cv2.addWeighted(src1=np.array(b2, dtype=float), alpha=float(ALPHA), src2=np.array(wm_pixel, dtype=float), beta=float(BETA), gamma=gamma)
        if wm1 == round(dst1[0][0]) and wm2 == round(dst2[0][0]):
            print(f"{wm_pixel} {ALPHA} {gamma}")
