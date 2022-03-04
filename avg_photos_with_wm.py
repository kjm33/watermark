from pathlib import Path
from collections import Counter
import numpy as np
import cv2

input_dir = Path('~/Pictures/sesja_natka/sesja_Natka/').expanduser()

img_paths = input_dir.glob("*.jpg")

img_by_name = {str(path.name): cv2.imread(str(path)) for path in img_paths}

most_common_resolution = Counter([img.shape for img in img_by_name.values()]).most_common(1)[0][0]

imgs_with_most_popular_shape = [img for img in img_by_name.values() if img.shape == most_common_resolution]
avg_img = np.mean(imgs_with_most_popular_shape, axis=0)

cv2.imwrite(str(input_dir / "avg_img.jpg"), avg_img)
