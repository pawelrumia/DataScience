from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

y1 = random.normal(loc=1, scale=2, size=5)
y2 = random.binomial(n=10, p=0.2, size=10)
y3 = random.poisson(lam=2, size=10)
y4 = random.uniform(size=(3, 3))
y5 = random.logistic(loc=1, scale=2, size=5)

print(y1)
print(y2)
print(y3)
print(y4)
print(y5)

plt.figure(figsize=(15, 15))
plt.subplot(331)
sns.distplot(random.normal(size=1000), hist=False)
plt.title('Normal Distribution')
plt.show()