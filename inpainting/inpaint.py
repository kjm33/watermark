import cv2
from common.watermark import get_alpha

mask = get_alpha()
radius = 3

# img = cv2.imread("../watermarks/DSC_8511_8.png")
img = cv2.imread("../watermarks/DSC_8752_5.png")
output = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS

cv2.imshow("original", img)
cv2.imshow("inpainted", output)

cv2.waitKey()
