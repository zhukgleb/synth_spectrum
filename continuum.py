import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def spectrum_finder(folder_data: list) -> list:
    spectrum_list = []
    for i in range(len(folder_data)):
        if str(folder_data[i]).find(".spec") != -1:
            spectrum_list.append(str(folder_data[i]))
    return spectrum_list

def data_graber(path2spectrum: Path) -> np.ndarray:
    data = np.genfromtxt(path2spectrum, comments="#")
    return data

def read_spectrum_grid(path2grid: Path) -> list:
    folder_data = os.listdir(path2grid)
    spectrum_list = spectrum_finder(folder_data)

    data = []
    for i in range(len(spectrum_list)):
        data.append(data_graber(path2grid / spectrum_list[i]))
    return data

def deriv_flux(normal_flux: np.ndarray) -> np.ndarray:
    delta_arr = []
    delta = 2
    for i in range(1, len(normal_flux)):
        delta_arr.append((abs(normal_flux[i] - normal_flux[i - 1])) / delta)
        
    delta_arr.append(1)
    return delta_arr

def deriv_diff_map():
    pass

def vizz(path2grid):
    data = read_spectrum_grid(path2grid)
    test_data = data[0]
    d_flux = deriv_flux(test_data[:, 1])
    d_wave = test_data[:, 0][1:]
    # plt.plot(d_wave, d_flux)
    fig, ax = plt.subplots()
    sc = ax.scatter(test_data[:, 0], test_data[:, 1], c=d_flux, cmap="plasma")
    plt.colorbar(sc, label="Derivative")
    # for i in range(len(data)):
    #     plt.plot(data[i][:, 0], data[i][:, 1])
    plt.show()


if __name__ == "__main__":
    vizz(Path("data/2025-10-14-01-08-56_0.22393946992408553_NLTE_synthetic_spectra_parameters/"))
