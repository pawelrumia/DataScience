import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 4, 2]

# plt.plot(x, y)
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')

# plt.show()

print("--------------------------")
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 8, 38]
# plt.bar(categories, values)

plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot')
plt.savefig('training.png')
# plt.show()

print("--------------")
x = [10, 20, 30, 40, 50]
y = [20, 40, 20, 60, 40]

# plt.scatter(x, y)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')

print("-------------------------")
labels = ['A', 'B', 'C', 'D']
sizes = [150, 300, 250, 200]
explode = [0, 0.1, 0, 0]

plt.pie(sizes, labels=labels, startangle=10, autopct='%1.2f%%', explode=explode)
plt.legend(title='Numbers')
plt.savefig('training.png')
plt.title('Pie Chart Example')
plt.show()

print("-------------------------------")
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
values = [10, 25, 15, 30]

# Creating a horizontal bar chart
# plt.barh(categories, values)

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Chart')

# plt.show()

categories = ['Category 1', 'Category 2', 'Category 3']
values1 = [3, 5, 2]
values2 = [1, 2, 4]

# plt.bar(categories, values1, label='Group 1')
# plt.bar(categories, values2, bottom=values1, label='Group 2')

plt.xlabel('-------Categories--------')
plt.ylabel('Values')
plt.title('Stacked Bar Chart')
plt.legend()

# plt.show()
