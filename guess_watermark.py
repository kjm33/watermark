from pathlib import Path
import numpy as np
import cv2


def mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()


ITERATIONS = 800

dst = cv2.imread("./samples/center_cross_25_x_25.png", 0)
dst_mode = 244

background = np.full(dst.shape, dst_mode, dtype=np.uint8)

np.random.seed(42)
watermark_weights = np.random.rand(*dst.shape)
gamma = 0.0

alpha = 0.01

for _ in range(ITERATIONS):
    watermark = watermark_weights*255  # matrix of floats
    watermark = watermark.astype(np.uint8)  # back to uchar

    trans_ratio = 0.8  # let's use first element as an alpha ratio
    trans_ratio_neg = 1.0 - trans_ratio  # beta

    blended = cv2.addWeighted(src1=background, alpha=trans_ratio, src2=watermark, beta=trans_ratio_neg, gamma=gamma)

    print(mse(dst, blended))

    delta = dst - blended

    delta_weights = delta / 255

    watermark_weights -= delta_weights*alpha

cv2.imwrite(str(Path("out") / f"guessed_wm_{ITERATIONS}_iters.jpg"), watermark)
