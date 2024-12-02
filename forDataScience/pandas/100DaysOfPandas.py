import pandas as pd
import numpy as np

arr = [10, 20, 30, 40]
series = pd.Series(arr, index=['a', 'b', 'c', 'd'])
print(series)

# creating a dataframe
data = {
    "age": [19, 33, 29],
    "waga": [94, 59, 76]
}


def dodawanie(x):
    return x + x


dataframe = pd.DataFrame(data)
print(dataframe)

print("==================")
a = np.array([1, 2, 3, 4])
b = np.array(['A', 'B', 'C', 'D'])
c = np.array(['Kop', 'San', 'Sat', 'Pune'])

newData = {'number': a, 'letter': b, 'something': c}
newDF = pd.DataFrame(newData)
print(newDF)
print(f"first row: {newDF.loc[1:2]}")
print(newDF['something'])
filtered = newDF[newDF['number'] >= 2]
print(f'filtered: {filtered}')
newDF['Imie'] = ['Patison', 'Jasio', 'Ja', 'Babcia']

print(newDF)

# drukowanie row
print(newDF.loc[2])

sorted = newDF.sort_values(by='Imie')
print(sorted)

print("---------------------------")
newDF['dodawane'] = newDF['number'].apply(dodawanie)
print(newDF)

category_mapping = {'A': 'super', 'B': 'niezle', 'C': 'bywalo lepiej', 'D': 'do poprawy'}
newDF['category_mapped'] = newDF['letter'].map(category_mapping)
print(newDF)
