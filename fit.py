from scipy.optimize import curve_fit
from numpy import exp as n_exp
from numpy import where


def gaussian_function(x, amplitude, mean, sigma, shift):
    return amplitude * n_exp(-(x - mean)**2 / (2 * sigma**2)) + shift


def gauss_corr_fit(corr_arr: list, z: float, sigma: float, shift: float):
    plot = False
    lag = corr_arr[0] 
    corr = corr_arr[1]
    wings_t = 25000  # wings in m/s
    corr_crop = where((lag*299792458>= z-wings_t) & (lag*299792458 <= z+wings_t))
    lag = lag[corr_crop]
    corr = corr[corr_crop]
    initial_guess = [max(corr), z, sigma, shift]
    fit_params, covariance = curve_fit(gaussian_function, 
                                       lag*299792458, corr,
                                       p0=initial_guess)
    del covariance  # Just for fun
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(lag*299792458, gaussian_function(lag*299792458, *fit_params), 'r-', label='fit')
        plt.plot(lag*299792458, corr)
        plt.xlabel("Correlation speed, m/s")
        plt.ylabel("Correlation Signal")
        plt.show()

    return fit_params[2]
