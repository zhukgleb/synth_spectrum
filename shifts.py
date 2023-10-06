from data import extract_data, get_path2 
import numpy as np



def doppler_shift(v: float, lambda_0=5000):
    # speed of light in meters in seconds
    c = 299792458
    delta_lambda = lambda_0 * (v / c)
    return delta_lambda


def make_shift(data: np.ndarray, velocity: float):
    data = data + doppler_shift(velocity)
    pass

if __name__ == "__main__":
    p = get_path2("Test_spectrum.syn")
    ang_1, flux_1, _ = extract_data(p)
    ang_2, flux_2, _ = extract_data(p)

    start_ang = 4600 
    end_ang = 6400

    index_start_1 = np.where(ang_1==start_ang)[0][0]
    index_end_1 = np.where(ang_1==end_ang)[0][0]
    ang_1 = ang_1[index_start_1:index_end_1]
    flux_1 = flux_1[index_start_1:index_end_1]

    index_start_2 = np.where(ang_2==start_ang)[0][0]
    index_end_2 = np.where(ang_2==end_ang)[0][0]
    ang_2 = ang_2[index_start_2:index_end_2]
    flux_2 = flux_2[index_start_2:index_end_2]

    print(doppler_shift(5*1000))
