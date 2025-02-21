import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.signal import find_peaks
import scienceplots


def load_spectra(template_path, observed_path, wavelength_min, wavelength_max):
    template = np.loadtxt(template_path)  # [λ, I]
    observed = np.loadtxt(observed_path)  # [λ, I]

    mask = (observed[:, 0] >= wavelength_min) & (observed[:, 0] <= wavelength_max)
    observed = observed[mask]
    template = template[
        (template[:, 0] >= wavelength_min) & (template[:, 0] <= wavelength_max)
    ]

    for i in range(len(observed)):
        if observed[:, 1][i] > 1:
            observed[:, 1][i] = 1

    return template, observed


def interpolate_template(observed, template):
    return np.interp(observed[:, 0], template[:, 0], template[:, 1])


def compute_bf_for_segment(segment_indices, observed, interp_template, v_grid, c):
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

    segment_matrix /= np.linalg.norm(segment_matrix, axis=1, keepdims=True) + 1e-10

    try:
        bf_segment, _ = nnls(segment_matrix.T, segment_observed)
    except RuntimeError:
        bf_segment = np.full(len(v_grid), np.nan)

    return bf_segment


def calculate_bf_map(observed, interp_template, v_grid, c, segment_width):
    num_segments = len(observed[:, 0]) // segment_width
    segment_indices_list = [
        slice(i, min(i + segment_width, len(observed[:, 0])))
        for i in range(0, len(observed[:, 0]), segment_width)
    ]

    bf_map = np.zeros((len(v_grid), len(observed[:, 0])))

    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(
            tqdm(
                executor.map(
                    compute_bf_for_segment,
                    segment_indices_list,
                    [observed] * len(segment_indices_list),
                    [interp_template] * len(segment_indices_list),
                    [v_grid] * len(segment_indices_list),
                    [c] * len(segment_indices_list),
                ),
                total=len(segment_indices_list),
                desc="Calculating BF",
                unit="segment",
            )
        )

    for idx, bf_segment in enumerate(results):
        start = idx * segment_width
        end = start + segment_width

        if end > bf_map.shape[1]:
            end = bf_map.shape[1]
            bf_map[:, start:end] = np.tile(bf_segment[:, None], (1, end - start))
        else:
            bf_map[:, start:end] = np.tile(bf_segment[:, None], (1, segment_width))

    return bf_map


def plot_bf_map(bf_map, observed, v_grid):
    plt.figure(figsize=(12, 6))
    extent = [observed[:, 0].min(), observed[:, 0].max(), v_grid.min(), v_grid.max()]
    plt.imshow(bf_map, aspect="auto", origin="lower", extent=extent, cmap="plasma")
    plt.colorbar(label="BF Amplitude")
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Radial Velocity (km/s)")
    plt.title("Broadening Function Map")
    plt.show()


def calculate_bf_profile(observed, interp_template, v_grid, c):
    broadening_matrix = np.zeros((len(v_grid), len(observed)))
    for i, v in enumerate(v_grid):
        shift_factor = 1 + v / c
        shifted_template = np.interp(
            observed[:, 0] * shift_factor,
            observed[:, 0],
            interp_template,
            left=0,
            right=0,
        )
        broadening_matrix[i] = shifted_template

    bf_profile, _ = nnls(broadening_matrix.T, observed[:, 1])
    return bf_profile


def plot_bf_profile(v_grid, bf_profile):
    with plt.style.context("science"):
        plt.plot(v_grid, bf_profile, color="black", alpha=0.8)
        plt.xlabel("Radial Velocity (km/s)")
        plt.ylabel("BF Amplitude")
        plt.title("Broadening Function")
        plt.grid()
        plt.show()


def find_top_bf_peaks(bf_map, v_grid, top_n=3, min_distance_kms=10):
    mean_bf = np.nanmean(bf_map, axis=1)
    velocity_step = np.abs(v_grid[1] - v_grid[0])
    min_distance_indices = int(np.round(min_distance_kms / velocity_step))

    peak_indices, _ = find_peaks(mean_bf, distance=min_distance_indices)

    if len(peak_indices) == 0:
        raise ValueError("Не удалось найти пики в BF.")

    sorted_peaks = sorted(peak_indices, key=lambda idx: mean_bf[idx], reverse=True)
    selected_peaks = sorted_peaks[:top_n]

    return v_grid[selected_peaks], mean_bf[selected_peaks]


def plot_bf_peaks(observed, bf_map, v_grid, top_velocities):
    with plt.style.context("science"):
        plt.figure(figsize=(12, 8))
        for i, v in enumerate(top_velocities):
            velocity_idx = (np.abs(v_grid - v)).argmin()
            bf_layer = bf_map[velocity_idx, :]
            bf_scaled = bf_layer / np.nanmax(bf_layer) * np.max(observed[:, 1])

            plt.subplot(3, 1, i + 1)
            plt.plot(
                observed[:, 0], observed[:, 1], label="Observed Spectrum", color="black"
            )
            plt.plot(
                observed[:, 0],
                bf_scaled,
                label=f"BF Layer at {v:.1f} km/s",
                color="crimson",
                linestyle="--",
            )
            plt.xlabel("Wavelength (Å)")
            plt.ylabel("Intensity")
            plt.title(f"Observed Spectrum with BF Layer at {v:.1f} km/s")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()


def main():
    template_path = "/Users/beta/synth_spectrum/iras07430.txt"
    observed_path = "/Users/beta/synth_spectrum/iras2020.txt"
    wavelength_min, wavelength_max = 4700, 5900
    c = 299792.458  # скорость света, км/с
    v_grid = np.linspace(-20, 20, 300)  # сетка скоростей, км/с
    segment_width = 8

    template, observed = load_spectra(
        template_path, observed_path, wavelength_min, wavelength_max
    )
    interp_template = interpolate_template(observed, template)
    bf_map = calculate_bf_map(observed, interp_template, v_grid, c, segment_width)
    plot_bf_map(bf_map, observed, v_grid)

    bf_profile = calculate_bf_profile(observed, interp_template, v_grid, c)
    plot_bf_profile(v_grid, bf_profile)

    top_n_peaks = 2
    min_distance_kms = 2
    top_velocities, top_amplitudes = find_top_bf_peaks(
        bf_map, v_grid, top_n=top_n_peaks, min_distance_kms=min_distance_kms
    )
    plot_bf_peaks(observed, bf_map, v_grid, top_velocities)


if __name__ == "__main__":
    main()

