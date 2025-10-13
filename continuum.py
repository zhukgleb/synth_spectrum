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


def read_spectrum_grid(path2grid: Path) -> np.ndarray:
    folder_data = os.listdir(path2grid)
    spectrum_list = spectrum_finder(folder_data)
    print(spectrum_list)
    return np.ndarray([0])


if __name__ == "__main__":
    read_spectrum_grid(Path("data/2025-10-14-01-08-56_0.22393946992408553_NLTE_synthetic_spectra_parameters/"))
