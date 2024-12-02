import numpy as np
import matplotlib.pyplot as plt

def simulate_data(end: int, how_many: int) -> np.ndarray:
    return np.linspace(0, end, how_many, endpoint=False)

def simulate_3d(
    num_dimensions: int = 3, last_number: int = 100, points_per_dimension: int = 10
) -> np.ndarray:
    grids = [simulate_data(last_number, points_per_dimension) for _ in range(num_dimensions)]
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.array(mesh).reshape(num_dimensions, -1)

def plot_3d(data: np.ndarray) -> bool:
    if data.shape[0] != 3:
        return False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(*data)
    plt.show()
    return True

# Przykład użycia:
data = simulate_3d(3, 10, 5)
plot_3d(data)