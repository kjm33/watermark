from pathlib import Path
import cv2
import numpy as np
from functools import reduce

BLACK = (0, 0, 0)

# https://stackoverflow.com/a/50900143
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


input_dir = Path("./examples")
examples_paths = input_dir.glob("*.png")

bin_examples = []


for example_path in examples_paths:
    example_img = cv2.imread(str(example_path))
    img = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
    most_common_pixel = unique_count_app(img)
    img[np.all(img == most_common_pixel, axis=2)] = BLACK
    thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bin_examples.append(thresh)


mask = reduce(cv2.bitwise_and, bin_examples)

cv2.imshow("mask", mask)
cv2.imwrite("common_mask.png", mask)
cv2.waitKey()
