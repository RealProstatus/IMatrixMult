import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = np.loadtxt("matrix_results.txt")
sizes = data[:, 0].astype(int)  # размеры матриц (N)
gflops = data[:, 1]
bandwidth = data[:, 2]

# Переводим в мегабайты (3 матрицы по N*N double)
size_MB = 3 * (sizes ** 2) * 8 / (1024 ** 2)

# Построение графиков
plt.figure(figsize=(12, 8))
plt.plot(size_MB, gflops, marker='o', label="Performance")
plt.plot(size_MB, bandwidth, marker='x', label="Bandwidth")

# Оформление графика с увеличенными шрифтами
plt.xlabel("Total Data Size (MB)", fontsize=18)
plt.ylabel("Performance (GFLOPS) / Bandwidth (GB/s)", fontsize=18)
plt.title("Matrix Multiplication Performance vs Data Size", fontsize=18)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend(fontsize=18)

plt.xscale("log")
plt.yscale("log")

# Увеличим размер шрифта для делений на осях
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()
plt.show()
