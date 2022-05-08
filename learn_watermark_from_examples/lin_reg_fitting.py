from sklearn.linear_model import LinearRegression
import numpy as np

cols = range(1, 264)
PIXEL_RANGE = np.array(range(50, 70)).reshape(1, -1)
rois_206 = np.loadtxt("./206_column.csv", delimiter=',', usecols=cols)

for row in rois_206:
    pixels_around_wm = row[50:70]
    reg = LinearRegression().fit(PIXEL_RANGE.T, pixels_around_wm.reshape(1, -1).T)
    a = reg.coef_[0][0]
    b = reg.intercept_[0]
    print(f"a={a:.2f}, b={b:.2f}")
print()
