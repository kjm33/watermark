from pathlib import Path
import matplotlib.pyplot as plt
#from fastai.data.block import DataBlock
#from fastai.data.transforms import get_image_files, RandomSplitter
#from fastai.vision.data import ImageBlock
#from fastai.vision.learner import cnn_learner, unet_learner
#from fastai.vision.models.all import *
#from fastai.metrics import error_rate
#from fastai.losses import MSELossFlat
from fastai.vision.all import *
# https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/blob/master/Computer%20Vision/05_Style_Transfer.ipynb#DataLoaders-and-Learner

clear_images = Path("../examples/cut_images/")
blended_images = Path("../examples/blended/")

dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=get_image_files,
                   get_y=lambda img: clear_images/img.name,
                   splitter=RandomSplitter(0.2, seed=42))

dls = dblock.dataloaders(blended_images, bs=32)
# dls.valid.show_batch(max_n=3, nrows=1)
# plt.show()

loss_gen = MSELossFlat()
learner = unet_learner(dls, resnet34, n_out=3, loss_func=loss_gen)
learner.fit_tune(1)
learner.show_results(rows=3)
print("zuo")

