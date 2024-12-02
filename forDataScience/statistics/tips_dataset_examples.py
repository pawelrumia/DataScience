import numpy as np
import pandas as pd
import statistics
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')
print(df.head())
df_numeric = pd.get_dummies(df)
print(df_numeric)
correlation_matrix = df_numeric.corr()
print(correlation_matrix)
# sns.pairplot(correlation_matrix)
# plt.show()