from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

sample_path = Path("./watermarks/DSC_8767_5.png")
sample = cv2.imread(str(sample_path))
MODE_COLOR = (244,)*3

alpha = 0.85
beta = 0.15
background = np.full(sample.shape, MODE_COLOR[0], dtype=float)

bckg_ratio = alpha/beta
watermark = sample.astype(float)/beta - bckg_ratio*background
watermark_single_channel = watermark[:, :, 0]

max_pixel = watermark_single_channel.max()
gamma = int(math.ceil(max_pixel) - 255)  # 63

watermark_normalized = cv2.convertScaleAbs(watermark)

cv2.imshow("watermark scaled", watermark_normalized)
# cv2.imwrite("watermark_calculated.png", watermark_normalized)
cv2.waitKey()

# TODO: addWeighted(background, calculated_wm, w/ params) <- compare by ssim with sample image
