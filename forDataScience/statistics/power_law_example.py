import numpy as np
import seaborn as sns
from scipy.stats import pareto, expon, norm, uniform
import matplotlib.pyplot as plt

x = np.linspace(0.1, 10, 1000)
alpha = 2.5
lambda_exp = 0.5
mu = 0
sigma = 1
a = 2
b = 8


pareto = pareto.pdf(x, alpha, scale=1)
exp = expon.pdf(x, scale=1 / lambda_exp)
norm = norm.pdf(x, mu, sigma)
uniform = uniform.pdf(x, a, b - a)

# plot for each distribution
plt.figure(figsize=(10, 6))

plt.plot(x, pareto, label='pareto', color='b')
plt.plot(x, exp, label='exponential', color='g')
plt.plot(x, norm, label='normal', color='r')
plt.plot(x, uniform, label='uniform', color='m')

plt.title("Power law distribution")
plt.xlabel("x")
plt.ylabel("density")
plt.legend()
plt.grid(True)
plt.show()
