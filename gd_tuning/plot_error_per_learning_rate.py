import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logs_dir = Path('error_logs/')

log_paths = logs_dir.glob("errors*txt")

"""
t = np.linspace(0, 1)
y1 = 2 * np.sin(2*np.pi*t)
y2 = 4 * np.sin(2*np.pi*2*t)

fig, ax = plt.subplots()
ax.set_title('Click on legend line to toggle line on/off')
line1, = ax.plot(t, y1, lw=2, label='1 Hz')
line2, = ax.plot(t, y2, lw=2, label='2 Hz')
leg = ax.legend(fancybox=True, shadow=True)
"""

fig, ax = plt.subplots()

for log_path in log_paths:
    basename = log_path.stem  # 'errors_a_0_01'
    alpha_str = basename.split('_a_')[-1].replace('_', '.')
    alpha = float(alpha_str)  # 0.01
    print(f"{basename} -> {alpha}")
    data_series = np.loadtxt(log_path)
    ax.plot(data_series, label=alpha)

plt.legend()
# TODO: legend and colors look poorly - move legend outside the plotting area
plt.show()

