import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris.columns)
print(iris.head())
# sns.histplot(data=iris)
# plt.show()

print("--------------")
print(iris['species'].unique())


# sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', size='petal_length', sizes=(10, 150), data=iris)
# sns.lineplot(x="sepal_length", y='sepal_width', data = iris)
# sns.kdeplot(data=iris, x="sepal_length",hue="species", fill=False, alpha=0.6, linewidth=1.5)
data = sns.load_dataset('tips')
# sns.regplot(x="total_bill", y="tip", data=data)
sns.relplot(x="total_bill", y="tip", hue="day", data=data, kind="scatter")
plt.savefig('training.png')
plt.show()
