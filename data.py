import numpy as np
import matplotlib.pyplot as plt
from os import path as p
# Simply data extraction from column-like file
# return a angstroms, flux and continuum


def extract_data(path: str):
    data = np.genfromtxt(path)
    ang = data[:, 0]
    red_flux = data[:, 1]
    cont = data[:, 3]
    ang.astype(float)
    red_flux.astype(float)
    return [ang, red_flux, cont]


# Return a close array-value to value 
def find_nearest(array: np.ndarray, value: float):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Temp function to define a path to files
def get_path2(filename):
    path2exe = str(p.abspath("gui.py"))
    path2data = path2exe.replace("gui.py", "")
    path2data = path2data + "data/" + filename
    return path2data

if __name__ == "__main__":
    x ,y, _ = extract_data("data/Na5889.syn")
    plt.plot(x, y)
    plt.show()
