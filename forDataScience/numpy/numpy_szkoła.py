from timeit import timeit
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np


def compare_operations(N: int, iters: int) -> Tuple[float, float, float]:
    lista = list(range(N))
    tablica = np.arange(N)

    # Dodawanie
    list_add_time = timeit(lambda: [el + 1 for el in lista], number=iters)
    numpy_add_time = timeit(lambda: tablica + 1, number=iters)

    # Mnożenie
    list_mul_time = timeit(lambda: [el * 2 for el in lista], number=iters)
    numpy_mul_time = timeit(lambda: tablica * 2, number=iters)

    # Podnoszenie do kwadratu
    list_square_time = timeit(lambda: [el ** 2 for el in lista], number=iters)
    numpy_square_time = timeit(lambda: tablica ** 2, number=iters)

    # Obliczenie średniej proporcji czasu wykonania operacji
    add_ratio = list_add_time / numpy_add_time
    mul_ratio = list_mul_time / numpy_mul_time
    square_ratio = list_square_time / numpy_square_time

    return round(add_ratio, 2), round(mul_ratio, 2), round(square_ratio, 2)


N = 1_000
iters_range = range(1, 21)
add_ratios, mul_ratios, square_ratios = [], [], []

for iters in iters_range:
    add_ratio, mul_ratio, square_ratio = compare_operations(N, iters)
    add_ratios.append(add_ratio)
    mul_ratios.append(mul_ratio)
    square_ratios.append(square_ratio)

plt.figure(figsize=(12, 6))
plt.plot(iters_range, add_ratios, label="Dodawanie (List vs NumPy)", marker='o')
plt.plot(iters_range, mul_ratios, label="Mnożenie (List vs NumPy)", marker='s')
plt.plot(iters_range, square_ratios, label="Kwadrat (List vs NumPy)", marker='^')

plt.title("Porównanie czasu wykonania operacji (List vs NumPy)")
plt.xlabel("Liczba powtórzeń (iters)")
plt.ylabel("Proporcja czasu (List / NumPy)")
plt.legend()
plt.grid()
plt.show()
