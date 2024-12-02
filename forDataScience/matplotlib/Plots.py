import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\DATA.csv')

# plt.boxplot(df['Weight'])
plt.pie(df['CO2'], startangle=90)
plt.show()