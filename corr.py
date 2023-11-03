import numpy as np
import matplotlib.pyplot as plt
from specutils.analysis import template_correlate
from specutils import Spectrum1D
from astropy import units as u
from astropy.nddata import StdDevUncertainty 
from scipy.signal import correlate
from scipy.signal import correlation_lags


start_ang = 4600
end_ang = 5400
velocity = 1000


def correlation(series_1: np.ndarray, series_2: np.ndarray):
    shortest = min(series_1.shape[0], series_2.shape[0])
    return np.corrcoef(series_1[:shortest], series_2[:shortest])[0, 1]


def plot_correlation(series_1: np.ndarray, series_2: np.ndarray, text: str):
    plt.figure(figsize=(10, 8))
    plt.plot(series_1, label="First spectrum")
    plt.plot(series_2, label="Second spectrum")
    plt.title(f"Correlation {text}: {correlation(series_1, series_2)}")
    plt.legend(loc="best")
    plt.show()


def shift_for_maximum_correlation(series_1: np.ndarray, series_2: np.ndarray):
    correlation_result = correlate(series_1, series_2, mode="full")
    lags = correlation_lags(series_1.size, series_2.size, mode="full")
    lag = lags[np.argmax(correlation_result)]

#    lag_corr_arr = np.column_stack((lags, correlation_result))
#    sortd_lag_corr_arr = lag_corr_arr[lag_corr_arr[:, 1].argsort()]
#    plt.title("Lag-correlation")
#    plt.plot(sortd_lag_corr_arr[:, 0], sortd_lag_corr_arr[:, 1])
#    plt.show()

    print(f"Best lag: {lag}")
    if lag < 0:
        series_2 = series_2[-lag:]
    else:
        series_1 = series_1[lag:]
    return series_1, series_2, lag


def calculate_correlation(spectrum, template_spectrum, method: str="astropy"):
    lag = 0
    correlation = []
    ang_resolution = spectrum[0][1] - spectrum[0][0]
    if method is "astropy":
        # Astropy Correlation
        spec_unit = u.erg / u.s / (u.cm * u.cm) / u.AA
        uncer = StdDevUncertainty(0.2*np.ones(spectrum[1].shape)*spec_unit)

        spectrum = Spectrum1D(spectral_axis=spectrum[0]*u.AA,
                                  flux=spectrum[1]*spec_unit,
                                  uncertainty=uncer)
        template_spectrum = Spectrum1D(spectral_axis=template_spectrum[0]*u.AA,
                                       flux=template_spectrum[1]*spec_unit,
                                       uncertainty=uncer)
        correlate, lags = template_correlate(spectrum, template_spectrum,
                                             lag_units=u.dimensionless_unscaled)
        lag_corr_arr = np.column_stack((lags, correlate))
        sortd_lag_corr_arr = lag_corr_arr[lag_corr_arr[:, 1].argsort()]
        lag, correlation = sortd_lag_corr_arr[0], sortd_lag_corr_arr[1]

    elif method is "scipy":
        spectrum_flux = spectrum[1] 
        template_flux = template_spectrum[1]

        shifted_spectrum, shifted_template, index_lag = shift_for_maximum_correlation(spectrum_flux, template_flux)
        plot_correlation(shifted_spectrum, shifted_template, text="after shifting")
        lag = calculate_doppler_from_shift(ang_resolution * index_lag)
        print(f"lag in meters is : {lag}")
        
 
    return lag, correlation


if __name__ == "__main__":
    from data import make_shifted_data
    a1, f1, a2, f2 = make_shifted_data(1000)
    lag, corr = calculate_correlation([a1, f1], [a2, f2], "scipy")
