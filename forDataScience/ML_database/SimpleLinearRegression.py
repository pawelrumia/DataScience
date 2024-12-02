import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

marketing = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\tvmarketing.csv')
# marketing.head()
# print(marketing.columns)
marketing.info()
print(marketing.shape)

plt.scatter(marketing['TV'], marketing['Sales'], color='green')
plt.show()