from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.watermark import get_alpha

sample_path = Path("../watermarks/DSC_8767_5.png")
sample = cv2.imread(str(sample_path))
MODE_COLOR = (244,)*3

alpha = 0.777997737556562  # calculated by linear regression
beta = 1 - alpha


background = np.full(sample.shape, MODE_COLOR[0], dtype=float)

bckg_ratio = alpha/beta
watermark = sample.astype(float)/beta - bckg_ratio*background
watermark_single_channel = watermark[:, :, 0]


# test = alpha*src + beta*wm -> src = test/alpha - beta/alpha*wm
# test = cv2.imread("../watermarks/DSC_8511_8.png")
# test = cv2.imread("../watermarks/DSC_8485_1.png")
test = cv2.imread("../watermarks/DSC_8500_10.png")
test_restored = test/alpha - (beta/alpha)*watermark
test_normalized = cv2.convertScaleAbs(test_restored)


mask = get_alpha()

merged = test.copy()
merged[np.where(mask == 255)] = test_normalized[np.where(mask == 255)]


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)

axs[0].set_title("original")
axs[0].imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))

axs[1].set_title('extracted WM')
axs[1].imshow(watermark_single_channel)

axs[2].set_title('restored')
axs[2].imshow(cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_BGR2RGB))

plt.show()


