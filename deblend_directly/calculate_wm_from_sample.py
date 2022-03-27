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

bckg_ratio = alpha/beta
watermark = sample.astype(float)/beta - bckg_ratio*background
watermark_single_channel = watermark[:, :, 0]


watermark_normalized = cv2.convertScaleAbs(watermark)

reblended = cv2.addWeighted(src1=background.astype(np.uint8), alpha=alpha, src2=watermark_normalized, beta=beta, gamma=0)


# test = alpha*src + beta*wm -> src = test/alpha - beta/alpha*wm
test = cv2.imread("./watermarks/DSC_8511_8.png")
test_restored = test/alpha - (beta/alpha)*watermark_normalized
test_normalized = cv2.convertScaleAbs(test_restored)

diff = test - test_restored
diff_single_channel = diff[:, :, 0]

plt.imshow(diff_single_channel)
plt.colorbar()
plt.title("Difference between original and restored images")
plt.show()

