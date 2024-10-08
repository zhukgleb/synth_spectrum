import numpy as np
from os import path as p


# Simply data extraction from column-like file
# return a angstroms, flux and continuum

def extract_data(path: str, text=False):
    data = np.genfromtxt(path)
    ang = data[:, 0]
    red_flux = data[:, 1]
    if text:
        return [ang, red_flux]
    cont = data[:, 3]
    flux = data[:, 2]
    ang.astype(float)
    red_flux.astype(float)
    return [ang, red_flux, flux]


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


# Return a simple test data. Tempo foo
def test_data(file_name: str = "Test_spectrum_med.syn", start: int=4500,
              end: int=5400):
    p = get_path2(file_name)
    ang, flux, _ = extract_data(p)
    index_start_1 = np.where(ang==start)[0][0]
    index_end_1 = np.where(ang==end)[0][0]
    ang = ang[index_start_1:index_end_1]
    flux = flux[index_start_1:index_end_1]

    return ang, flux

# Return a simple test data. Tempo foo
def test_data_ideal(file_name: str = "Test_spectrum.syn", start: int=4500,
              end: int=5400):
    p = get_path2(file_name)
    ang, flux, _ = extract_data(p)
    index_start_1 = np.where(ang==start)[0][0]
    index_end_1 = np.where(ang==end)[0][0]
    ang = ang[index_start_1:index_end_1]
    flux = flux[index_start_1:index_end_1]

    return ang, flux

# velocity in meters per second!
# raw function, need to re-write
# flux 1 is observed spectrum
# flux 2 is template spectrum


def test_data_template(file_name: str = "Test_spectrum.syn", start: int=4500,
              end: int=5400):
    p = get_path2(file_name)
    ang, flux, _ = extract_data(p)
    index_start_1 = np.where(ang==start)[0][0]
    index_end_1 = np.where(ang==end)[0][0]
    ang = ang[index_start_1:index_end_1]
    flux = flux[index_start_1:index_end_1]

    return ang, flux
