from pathlib import Path
import numpy as np
import cv2

# TODO: median filter -> contour detection, common shape and positions of centers

laplacian_normalized_out_path = Path("out") / "laplacian.jpg"
watermarks_gray = cv2.imread(str(laplacian_normalized_out_path), 0)
watermarks_median = cv2.medianBlur(watermarks_gray, 3)
thresh = 3
watermarks_bin = cv2.threshold(watermarks_median, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("median", watermarks_bin)
cv2.waitKey(0)
