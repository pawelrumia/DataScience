import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt

ages = [24, 34, 23, 25, 56, 76, 11, 35, 34, 37, 77, 73, 64, 69]

print(np.mean(ages))
print(statistics.mean(ages))

print(statistics.median(ages))
print(statistics.variance(ages))

df = sns.load_dataset('iris')
print(df.head())
sns.histplot(df['petal_length'], kde=True)
plt.show()