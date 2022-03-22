from pathlib import Path
import numpy as np
import cv2

# dst = alpha*src + beta*watermark -> guess SRC

EPS = 0.0001
iterations = 2000
trans_ratio = 0.8  # alpha is already used as a gradient descent learning rate
trans_ratio_neg = 0.2  # beta
# np.random.seed(42)
alpha = 0.02


def mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()


watermark = cv2.imread("../watermark.png", cv2.IMREAD_UNCHANGED)
watermark_alpha = watermark[:, :, 3]
watermark_single_channel = watermark[:, :, 0]  # R=G=B in our case

dst = cv2.imread("../watermarks/DSC_8511_8.png")
(dst_B, dst_G, dst_R) = cv2.split(dst)


def find_src_layer(dst_channel, watermark_channel):
    src_weights = dst_channel/255.0
    # TODO: copy masked pixels in watermark from dst
    wm_with_copied_bkg = watermark_channel.copy()

    # Copy pixel values of dst channel to watermark channel wherever the mask is white
    wm_with_copied_bkg[np.where(watermark_alpha == 255)] = dst_channel[np.where(watermark_alpha == 255)]
    wm_with_copied_bkg = wm_with_copied_bkg.astype(float)

    for _ in range(iterations):
        src = src_weights * 255  # matrix of floats

        blended = cv2.addWeighted(src1=src, alpha=trans_ratio, src2=wm_with_copied_bkg, beta=trans_ratio_neg, gamma=0)

        # desired error level < 0.0001
        error = mse(dst_channel, blended)
        # print(error)

        delta = dst_channel - blended
        delta_weights = delta / 255
        src_weights += delta_weights * alpha

        if error < EPS:
            break

    src = src_weights*255
    return src


src_B = find_src_layer(dst_B, watermark_single_channel)
src_G = find_src_layer(dst_G, watermark_single_channel)
src_R = find_src_layer(dst_R, watermark_single_channel)

src_rgb = [src_R, src_G, src_B]
src = cv2.merge(src_rgb, 3)
cv2.imwrite("src.png", src)
