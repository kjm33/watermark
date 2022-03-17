from pathlib import Path
import numpy as np
import cv2
import random as rng

# TODO: median filter -> contour detection, common shape and positions of centers

laplacian_normalized_out_path = Path("out") / "laplacian.jpg"
watermarks_gray = cv2.imread(str(laplacian_normalized_out_path), 0)
watermarks_median = cv2.medianBlur(watermarks_gray, 5)
thresh = 3
watermarks_bin = cv2.threshold(watermarks_median, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("median", watermarks_bin)


contours, hierarchy = cv2.findContours(watermarks_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# for cnt in contours:
    # x, y, w, h = cv2.boundingRect(cnt)

# from https://docs.opencv.org/4.x/df/d0d/tutorial_find_contours.html
rng.seed(42)
drawing = np.zeros((watermarks_bin.shape[0], watermarks_bin.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

cv2.imshow('Contours', drawing)

cv2.waitKey(0)
