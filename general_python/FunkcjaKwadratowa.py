import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x** 2 + 1

x_values = np.linspace(-10, 10, 200)  # 100 points from -10 to 10

# Compute corresponding y values
y_values = f(x_values)
plt.plot(x_values, y_values, label='f(x) = x**2 + 1', color='blue')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plot of f(x) = x**2 + 1")
plt.grid(True)
plt.legend()
plt.show()
