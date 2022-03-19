from pathlib import Path
import numpy as np
import cv2

laplacian_normalized_out_path = Path("out") / "laplacian_bin_Otsu.jpg"
watermarks_gray = cv2.imread(str(laplacian_normalized_out_path), 0)


contours, hierarchy = cv2.findContours(watermarks_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnt_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
true_watermarks = cnt_sorted[:15]

# y = 18, 318, 618 -> 3 rows
# x = 16, 316, 616, 916, 1216 -> 5 columns
# w = 264
# h = 264 :)

for idx, cnt in enumerate(true_watermarks, start=1):
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"{idx} {x} {y} {w} {h}")
print()
