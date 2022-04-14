import cv2
import numpy as np
from pathlib import Path
from common.watermark import _load, get_alpha, get_common_mask
from merge_image_with_watermark import blend_img_with_watermark
import random

MASKS = []


def load_masks():
    dir_with_masks = Path("../../learn_watermark_from_examples/")
    masks_paths = dir_with_masks.glob("common_mask_*png")

    masks = [cv2.imread(str(path)) for path in masks_paths]

    alpha1d = get_alpha()
    alpha3d = cv2.merge([alpha1d, alpha1d, alpha1d])
    masks.insert(1, alpha3d)

    return masks


def get_random_trans_rate():
    alpha10 = random.randrange(start=40, stop=80, step=5)
    alpha = alpha10/100
    return alpha


def get_random_mask():
    global MASKS
    if not MASKS:
        MASKS = load_masks()

    return random.choice(MASKS)


random.seed(42)
out_dir = Path("blended")
cut_images_paths = Path("./cut_images/").glob("*png")

watermark = _load()
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2BGR)

for cut_image_path in cut_images_paths:
    cut_image = cv2.imread(str(cut_image_path))
    mask = get_random_mask()
    alpha = get_random_trans_rate()

    blended = blend_img_with_watermark(img=cut_image, wm=watermark, mask=mask, alpha=alpha)

    cut_image_basename = cut_image_path.name
    blended_path = out_dir / cut_image_basename
    cv2.imwrite(str(blended_path), blended)

