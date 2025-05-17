import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
data = np.loadtxt("matrix_results.txt")
sizes = data[:, 0].astype(int)  # размеры матриц (N)
gflops = data[:, 1]
bandwidth = data[:, 2]

# Переводим в мегабайты (3 матрицы по N*N double)
size_MB = 3 * (sizes ** 2) * 8 / (1024 ** 2)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(size_MB, gflops, marker='o', label="GFLOPS")
plt.plot(size_MB, bandwidth, marker='x', label="Bandwidth (GB/s)")

plt.xlabel("Total Data Size (MB)")
plt.ylabel("Performance")
plt.title("Matrix Multiplication Performance vs Data Size")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()

# Логарифмическая шкала по оси X
plt.xscale("log")

# Если хочешь — можно и по Y:
# plt.yscale("log")

# Автоматическая подстройка делений по X
plt.tight_layout()
plt.show()
