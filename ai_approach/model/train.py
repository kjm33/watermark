from pathlib import Path
import matplotlib.pyplot as plt
from fastai.vision.all import *

clear_images = Path("../examples/cut_images/")
blended_images = Path("../examples/blended/")


def get_clear_image_by_path(blended_image: Path):
    return clear_images/blended_image.name


dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=get_image_files,
                   get_y=get_clear_image_by_path,
                   splitter=RandomSplitter(0.2, seed=42),
                   item_tfms=Resize(128)
                   )

dls = dblock.dataloaders(blended_images, bs=16)
# dls.valid.show_batch(max_n=3, nrows=1)
# plt.show()

loss_gen = MSELossFlat()
learner = unet_learner(dls, resnet34, n_out=3, loss_func=loss_gen)
learner.fine_tune()

