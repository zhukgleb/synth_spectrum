import numpy as np
import matplotlib.pyplot as plt

# Simply data extraction from column-like file
# return a angstroms, flux and continuum


def extract_data(path: str) -> [np.ndarray, np.ndarray, np.ndarray]:
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


if __name__ == "__main__":
    x ,y, _ = extract_data("data/Na5889.syn")
    plt.plot(x, y)
    plt.show()
