import numpy as np
import scipy.stats as stat
import pylab
import seaborn as sns
import matplotlib.pyplot as plt


data = np.random.normal(0.5, 0.2, 1000)
df = sns.load_dataset('iris')
# sns.histplot(data, kde=True)

def plot_data(sample):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(sample)
    plt.subplot(1, 2, 2)
    stat.probplot(sample, dist='norm', plot=pylab)
    plt.show()

# plot_data(data)
plot_data(df['sepal_widthk'])
