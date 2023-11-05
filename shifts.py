from data import extract_data, get_path2 
import numpy as np
from scipy.interpolate import interp1d
from data import test_data, test_data_ideal

def doppler_shift(v: float, lambda_0=5000):
    # speed of light in meters in seconds
    c = 299792458
    delta_lambda = lambda_0 * (v / c)
    return delta_lambda


def make_shift(ang: np.ndarray, velocity: float):
    ang_shift = doppler_shift(velocity)
    ang += ang_shift
    return ang 


def calculate_shift_index(ang_resolution: float, velocity: float):
    delta_lambda = doppler_shift(velocity)
    print(f"resolution is {ang_resolution}")
    print(f"delta lambda is {delta_lambda}")
    shift_index = delta_lambda // ang_resolution
    return int(shift_index)

def calculate_doppler_from_shift(ang_shift: float, lambda_0=5000):
    return ang_shift / lambda_0 * 299792458


def make_shifted_data(series_1, series_2, velocity: float):
    ang_1_original, flux_1_original = series_1[0], series_1[1]
    ang_2_original, flux_2_original = series_2[0], series_2[1]

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
    ang_2 = ang_2
    print(len(ang_1), len(flux_1), len(ang_1), len(flux_2))
    return ang_1, flux_1, ang_2, flux_2
    

if __name__ == "__main__":
    p = get_path2("Test_spectrum.syn")
    ang_1, flux_1, _ = extract_data(p)
    ang_2, flux_2, _ = extract_data(p)

    start_ang = 5000
    end_ang = 5001

    index_start_1 = np.where(ang_1==start_ang)[0][0]
    index_end_1 = np.where(ang_1==end_ang)[0][0]
    ang_1 = ang_1[index_start_1:index_end_1]
    flux_1 = flux_1[index_start_1:index_end_1]

    index_start_2 = np.where(ang_2==start_ang)[0][0]
    index_end_2 = np.where(ang_2==end_ang)[0][0]
    ang_2 = ang_2[index_start_2:index_end_2]
    flux_2 = flux_2[index_start_2:index_end_2]


    ang_2 = make_shift(ang_2, 10*1000)
