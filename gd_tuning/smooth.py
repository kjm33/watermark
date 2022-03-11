import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def smooth(data, window_length=100, polyorder=3):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


if __name__ == '__main__':
    data_orig = np.loadtxt("./error_logs/errors_a_0_02.txt")
    smoothed = smooth(data_orig)

    fig, ax = plt.subplots()

    ax.plot(data_orig, label='orig data')
    ax.plot(smoothed, label="smoothed")

    plt.legend()
    plt.show()
