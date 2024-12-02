import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zlib import crc32

# zmiana ustawien pandas odnosnie wyswietlania kolumn
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('C:\\Users\\Asus\\PycharmProjects\\DataScience\\various\\resources\\housing.csv')
print(df.info())
print("====================")
print(df.describe())
print(df.columns)
print(df.describe()[
          ['longitude', 'latitude', 'median_income', 'median_house_value', 'housing_median_age', 'total_rooms']])

print(df['ocean_proximity'].value_counts())

df.hist(bins=50, figsize=(20, 15))


# plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = df.reset_index() # Dodaje kolumnę `index`
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

train_set, test_set = split_train_test(df, 0.2)
print(len(train_set))
print(len(test_set))
housing_with_id["id"] = df["longitude"] * 1000 + df["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
print(len(train_set))
print(len(test_set))