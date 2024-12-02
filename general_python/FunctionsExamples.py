import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * 2 + 1


numbers = [0, 1, 2, 3, 4]
for number in numbers:
    print(f(number))

x_values = np.linspace(-10, 10, 200)  # 100 points from -10 to 10

# Compute corresponding y values
y_values = f(x_values)
plt.plot(x_values, y_values, label='f(x) = 2x + 1', color='blue')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plot of f(x) = 2x + 1")
plt.grid(True)
plt.legend()
plt.show()