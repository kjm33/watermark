import cv2
import numpy
import numpy as np

watermark = cv2.imread("watermark_w_bg.png")
b, g, r = cv2.split(watermark)
alpha = np.where(b == 244, 0, 255).astype(numpy.uint8)
rgba = [b, g, r, alpha]
dst = cv2.merge(rgba, 4)
# im = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)
cv2.imwrite("watermark.png", dst)
