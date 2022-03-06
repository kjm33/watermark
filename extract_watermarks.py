import cv2
from pathlib import Path

avg_img_path = Path("out") / "avg_img.jpg"
avg_img = cv2.imread(str(avg_img_path))
avg_img_gray = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(avg_img_gray, cv2.CV_64F)
laplacian_normalized = cv2.convertScaleAbs(laplacian)

cv2.imshow('normalized Lap', laplacian_normalized)


laplacian_normalized_out_path = Path("out") / "laplacian.jpg"

if not laplacian_normalized_out_path.exists():
    cv2.imwrite(str(laplacian_normalized_out_path), laplacian_normalized)

# watermarks are now barely visible (gray patterns on black background) so let's increase contrast with thresholding for
# further processing

# on the first sight foreground/background should be easy to separate with Otsu

otsu = cv2.threshold(laplacian_normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow("Otsu", otsu)
# bingo!

lap_bin_path = Path("out") / "laplacian_bin_Otsu.jpg"

if not lap_bin_path.exists():
    cv2.imwrite(str(lap_bin_path), otsu)


cv2.waitKey(0)

