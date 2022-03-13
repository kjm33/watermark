from pathlib import Path
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", type=int, default=800,
                        help="The number of iterations of GD")
    parser.add_argument("-a", "--alpha", type=float, default=0.01,
                        help="GD learning rate")
    return parser.parse_args()


def mse(img1, img2):
    return np.square(np.subtract(img1, img2)).mean()


args = get_params()
iterations = args.iterations

dst_orig = cv2.imread("./samples/center_cross_25_x_25.png", 0)
dst = dst_orig.astype(float)
dst_mode = 244.0

background = np.full(dst.shape, dst_mode, dtype=float)

np.random.seed(42)
watermark_weights = np.random.rand(*dst.shape)
gamma = 0.0

alpha = args.alpha
# TODO: test different values (simulated annealing?)

trans_ratio = 0.8  # alpha is already used as a gradient descent learning rate
trans_ratio_neg = 1.0 - trans_ratio  # beta

for _ in range(iterations):
    watermark = watermark_weights * 255  # matrix of floats
    # watermark = np.around(watermark)
    # watermark = watermark.astype(np.uint8)  # back to uchar
    # ^^ small changes in weights can be ignored/lost here due to casting (rounding?)

    """
    src1	first input array.
    alpha	weight of the first array elements.
    src2	second input array of the same size and channel number as src1.
    beta	weight of the second array elements.
    gamma	scalar added to each sum.
    """
    blended = cv2.addWeighted(src1=background, alpha=trans_ratio, src2=watermark, beta=trans_ratio_neg, gamma=gamma)
    blended_rounded = np.around(blended)
    blended_uchar = blended_rounded.astype(np.uint8)

    ssim, diff_img = structural_similarity(dst_orig, blended_uchar, full=True)

    print(mse(dst, blended), ssim)

    delta = dst - blended
    delta_weights = delta / 255
    watermark_weights += delta_weights * alpha

cv2.imwrite(str(Path("guessing_progress") / f"guessed_wm_{iterations}_iters.jpg"), watermark)
cv2.imwrite(str(Path("guessing_progress") / f"diff_{iterations}_iters.jpg"), diff_img)
