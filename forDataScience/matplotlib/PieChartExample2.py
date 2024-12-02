import pandas as pd
import matplotlib.pyplot as plt

data = [32, 45, 11, 23, 78, 29]
labels = ['Python', 'C', 'C++', 'PHP', 'Java', 'Ruby']
colors = ['saddlebrown', 'wheat', 'crimson', 'lightgrey','lightblue','darkblue']
plt.figure(figsize=(7,9))
plt.pie(data, labels=labels, explode=[0.0, 0, 0, 0, 0.08, 0], colors=colors, shadow=True, autopct='%.4f%%', startangle=180)
plt.legend()
plt.show()
