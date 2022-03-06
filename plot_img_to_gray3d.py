import numpy as np
import matplotlib.pyplot as plt
import cv2

# taken from https://stackoverflow.com/a/31806902

img = cv2.imread("./samples/single_watermark_on_smooth_bckg.png", 0)


# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

# show it
plt.show()
