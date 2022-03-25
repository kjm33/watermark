from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

sample_path = Path("./watermarks/DSC_8767_5.png")
sample = cv2.imread(str(sample_path))
MODE_COLOR = (244,)*3

alpha = 0.85
beta = 0.15
background = np.full(sample.shape, MODE_COLOR[0], dtype=float)

for alpha100 in range(85, 5, -5):
    alpha = alpha100/100.0
    beta = 1 - alpha

    bckg_ratio = alpha/beta
    watermark = sample.astype(float)/beta - bckg_ratio*background
    watermark_single_channel = watermark[:, :, 0]
    print(f"{alpha} max {watermark_single_channel.max()}")

