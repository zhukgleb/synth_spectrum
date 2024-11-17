import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from matplotlib.colors import LogNorm

# Загружаем спектры
template = np.loadtxt("/home/gamma/postagb.spec")  # [λ, I]
# observed = np.loadtxt(
#     "/home/gamma/TSFitPy/input_files/observed_spectra/iras2020.txt"
# )  # [λ, I]
observed = np.genfromtxt("binary_spectrum.txt")


wavelength_min, wavelength_max = 4700, 4800
mask = (observed[:, 0] >= wavelength_min) & (observed[:, 0] <= wavelength_max)
observed = observed[mask]
template = template[
    (template[:, 0] >= wavelength_min) & (template[:, 0] <= wavelength_max)
]


for i in range(len(observed)):
    if observed[:, 1][i] > 1:
        observed[:, 1][i] = 1

c = 299792.458  # скорость света, км/с
v_grid = np.linspace(-100, 100, 3000)  # сетка скоростей, км/с
segment_width = 3  # ширина сегмента спектра в пикселях

# Интерполяция шаблона
interp_template = np.interp(observed[:, 0], template[:, 0], template[:, 1])


# Функция для вычисления BF с обработкой ошибок
def compute_bf_for_segment(segment_indices):
    segment_observed = observed[segment_indices, 1]
    segment_matrix = np.zeros((len(v_grid), len(segment_observed)))

    for j, v in enumerate(v_grid):
        shift_factor = 1 + v / c
        shifted_template = np.interp(
            observed[segment_indices, 0] * shift_factor,
            observed[:, 0],
            interp_template,
            left=0,
            right=0,
        )
        segment_matrix[j, :] = shifted_template

    # Нормализация матрицы
    segment_matrix /= np.linalg.norm(segment_matrix, axis=1, keepdims=True) + 1e-10

    try:
        # Решение NNLS
        bf_segment, _ = nnls(segment_matrix.T, segment_observed)
    except RuntimeError:
        # Если ошибка, возвращаем NaN
        bf_segment = np.full(len(v_grid), np.nan)  # Или np.zeros(len(v_grid))
    return bf_segment


# Разбиение спектра на сегменты
num_segments = len(observed[:, 0]) // segment_width
segment_indices_list = [
    slice(i, min(i + segment_width, len(observed[:, 0])))
    for i in range(0, len(observed[:, 0]), segment_width)
]

# Параллельный расчёт BF с обработкой ошибок
bf_map = np.zeros((len(v_grid), len(observed[:, 0])))

with ThreadPoolExecutor(max_workers=24) as executor:
    results = list(
        tqdm(
            executor.map(compute_bf_for_segment, segment_indices_list),
            total=len(segment_indices_list),
            desc="Calculating BF",
            unit="segment",
        )
    )

# Объединение результатов в BF-карту
for idx, bf_segment in enumerate(results):
    start = idx * segment_width
    end = start + segment_width

    if end > bf_map.shape[1]:
        end = bf_map.shape[1]
        bf_map[:, start:end] = np.tile(bf_segment[:, None], (1, end - start))
    else:
        bf_map[:, start:end] = np.tile(bf_segment[:, None], (1, segment_width))

# Визуализация BF-карты с логарифмической шкалой
plt.figure(figsize=(12, 6))
extent = [observed[:, 0].min(), observed[:, 0].max(), v_grid.min(), v_grid.max()]
plt.imshow(bf_map, aspect="auto", origin="lower", extent=extent, cmap="plasma")
plt.colorbar(label="BF Amplitude")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Radial Velocity (km/s)")
plt.title("Broadening Function Map (Logarithmic Scale)")
plt.show()
