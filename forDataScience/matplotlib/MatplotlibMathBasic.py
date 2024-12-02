import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


fig, axes = plt.subplots(1, 3, figsize=(16,8))
ax1, ax2, ax3 = axes
x = np.linspace(0,10,500)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

ax1.set_ylabel('sin(x)', fontsize=30)
ax1.set_xlabel('x', fontsize=30)
ax2.set_ylabel('cos(x)', fontsize=30)
ax2.set_xlabel('x', fontsize=30)
ax3.set_ylabel('tan(x)', fontsize=30)
ax3.set_xlabel('x', fontsize=30)
fig.tight_layout(pad=2)

ax1.plot(x, y1, c='r', linewidth=4)
ax2.plot(x, y2, c='g', linewidth=4)
ax3.plot(x, y3, c='b', linewidth=4)
plt.show()