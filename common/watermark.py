from pathlib import Path
import cv2

WATERMARK_PATH = Path('~/Documents/projects/watermark/watermark.png').expanduser()
WATERMARK = None


def _load():
    global WATERMARK

    if not WATERMARK:
        WATERMARK = cv2.imread(str(WATERMARK_PATH), cv2.IMREAD_UNCHANGED)
    return WATERMARK


def get_alpha():
    watermark = _load()
    alpha = watermark[:, :, 3]
    return alpha
