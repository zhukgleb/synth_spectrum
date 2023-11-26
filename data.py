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

"""
def make_shifted_data(velocity: float):
    ang_1_original, flux_1_original = test_data()
    ang_2_original, flux_2_original = test_data_ideal()

    ds = doppler_shift(velocity)
    ang_resolution = ang_1_original[1] - ang_1_original[0]
    print(f"doppler shift as {ds}, lambda res is {ang_resolution}")
    interp_multi = round(ang_resolution / ds)
    dots = int(len(ang_2_original)*interp_multi) * 10
    ang_1 = np.linspace(np.min(ang_1_original), np.max(ang_1_original), dots)
    ang_2 = np.linspace(np.min(ang_2_original), np.max(ang_2_original), dots)
    f1 = interp1d(ang_1_original, flux_1_original, kind="quadratic")
    f2 = interp1d(ang_2_original, flux_2_original, kind="quadratic")
    flux_1 = f1(ang_1)
    flux_2 = f2(ang_2)

    ang_resolution = ang_1[1] - ang_1[0]
    shift_index = calculate_shift_index(ang_resolution, velocity)
    flux_2 = flux_2[shift_index:]
    
    return ang_1, flux_1, ang_2, flux_2
"""
