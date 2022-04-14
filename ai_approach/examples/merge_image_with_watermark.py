import cv2
from common.watermark import _load, get_alpha, get_common_mask
import numpy as np

INVISIBLE = (0, 0, 0)
VISIBLE = (255, 255, 255)


def blend_img_with_watermark(img, wm=None, mask=None, alpha=0.8):
    wm_with_filled_bg = wm.copy()
    wm_with_filled_bg[np.where(mask == INVISIBLE)] = img[np.where(mask == INVISIBLE)]

    beta = 1 - alpha
    gamma = 0

    blended = cv2.addWeighted(src1=img, alpha=alpha, src2=wm_with_filled_bg, beta=beta, gamma=gamma)

    return blended


if __name__ == '__main__':

    img = cv2.imread("./cut_images/...0539_11.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    watermark = _load()
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2BGR)

    alpha1d = get_alpha()
    alpha3d = cv2.merge([alpha1d, alpha1d, alpha1d])
    # mask = get_common_mask()
    mask = alpha3d

    blended = blend_img_with_watermark(img=img, wm=watermark, mask=mask, alpha=0.7)

    # cv2.imshow("wm filled alpha", wm_with_filled_bg)
    cv2.imshow("blended", blended)
    cv2.waitKey()

