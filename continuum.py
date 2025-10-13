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

def read_spectrum_grid(path2grid: Path) -> np.ndarray:
    folder_data = os.listdir(path2grid)
    spectrum_list = spectrum_finder(folder_data)

    data = []
    for i in range(len(spectrum_list)):
        data.append(data_graber(path2grid / spectrum_list[i]))
    return data


if __name__ == "__main__":
    read_spectrum_grid(Path("data/2025-10-14-01-08-56_0.22393946992408553_NLTE_synthetic_spectra_parameters/"))
