from sklearn.linear_model import LinearRegression
import numpy as np


def mse(a, b):
    return np.square(np.subtract(a, b)).mean()


cols = range(1, 264)
# PIXEL_RANGE = np.array(range(50, 70)).reshape(1, -1)
rois_206 = np.loadtxt("./206_column.csv", delimiter=',', usecols=cols)
model = LinearRegression()

for row in rois_206:
    pixels_around_wm = np.concatenate((row[41:51], row[71:81]), axis=0).reshape(1, -1).T
    pixel_range = np.array(range(1, len(pixels_around_wm)+1)).reshape(1, -1).T
    reg = model.fit(pixel_range, pixels_around_wm)
    a = reg.coef_[0][0]
    b = reg.intercept_[0]
    predicted_values = model.predict(pixel_range)
    error = mse(pixels_around_wm, predicted_values)
    print(f"a={a:.2f}, b={b:.2f} error={error:.2f}")
print()
