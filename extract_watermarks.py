import cv2
from pathlib import Path

avg_img_path = Path("out") / "avg_img.jpg"
avg_img = cv2.imread(str(avg_img_path))
avg_img_gray = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)

laplacian = cv2.Laplacian(avg_img_gray, cv2.CV_64F)
laplacian_normalized = cv2.convertScaleAbs(laplacian)

cv2.imshow('normalized Lap', laplacian_normalized)
cv2.waitKey(0)

laplacian_normalized_out_path = Path("out") / "laplacian.jpg"

if not laplacian_normalized_out_path.exists():
    cv2.imwrite(str(laplacian_normalized_out_path), laplacian_normalized)

