import numpy as np
import cv2


def mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()


dst = cv2.imread("./samples/center_cross_25_x_25.png", 0)
dst_mode = 244

background = np.full(dst.shape, dst_mode, dtype=np.uint8)

watermark_weights = np.random.rand(*dst.shape)
watermark = watermark_weights*255


cv2.imshow("watermark", watermark)
cv2.waitKey(0)

