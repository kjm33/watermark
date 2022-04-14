from pathlib import Path
import cv2

WATERMARK_PATH = Path('~/Documents/projects/watermark/watermark.png').expanduser()
WATERMARK = None
COMMON_MASK = None


def _load():
    global WATERMARK

    if WATERMARK is None:
        WATERMARK = cv2.imread(str(WATERMARK_PATH), cv2.IMREAD_UNCHANGED)
    return WATERMARK


def get_alpha():
    watermark = _load()
    alpha = watermark[:, :, 3]
    return alpha


def get_common_mask():
    global COMMON_MASK

    if not COMMON_MASK:
        common_mask_path = WATERMARK_PATH.parent / "./learn_watermark_from_examples/common_mask.png"
        COMMON_MASK = cv2.imread(str(common_mask_path))

    return COMMON_MASK
