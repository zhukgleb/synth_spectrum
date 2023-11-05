import numpy as np
import matplotlib.pyplot as plt
from specutils.analysis import template_correlate
from specutils import Spectrum1D
from astropy import units as u
from astropy.nddata import StdDevUncertainty 
from scipy.signal import correlate
from scipy.signal import correlation_lags
from shifts import calculate_doppler_from_shift


start_ang = 4500
end_ang = 5500
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


def calculate_correlation(spectrum_np, template_spectrum_np, method: str="astropy"):
    lag = 0
    correlation = []
    ang_resolution = spectrum_np[0][1] - spectrum_np[0][0]

    if method == "astropy":
        
        spectrum_sp = spectrum_np[0]
        spectrum_flux = spectrum_np[1]
        template_sp = spectrum_np[0]
        template_flux = spectrum_np[1]

        spec_unit = u.Unit('erg cm-2 s-1 AA-1') 
        shift_unit = u.AA / u.s
        unc_arr = [0.1 for x in range(len(spectrum_flux))]
        unc = StdDevUncertainty(unc_arr, unit=spec_unit)

        spectrum = Spectrum1D(spectral_axis=spectrum_sp*u.AA,
                                  flux=spectrum_flux*spec_unit, uncertainty=unc)
        template_spectrum = Spectrum1D(spectral_axis=template_sp*u.AA, 
                                       flux=template_flux*spec_unit)

        correlate, lags = template_correlate(spectrum, template_spectrum,
                                             lag_units=shift_unit)
        lag_corr_arr = np.column_stack((lags, correlate))
        sortd_lag_corr_arr = lag_corr_arr[lag_corr_arr[:, 1].argsort()]
        lag, correlation = sortd_lag_corr_arr[0], sortd_lag_corr_arr[1]

    elif method == "scipy":
        spectrum_flux = spectrum_np[1] 
        template_flux = template_spectrum_np[1]

        shifted_spectrum, shifted_template, index_lag = shift_for_maximum_correlation(spectrum_flux, template_flux)
        plot_correlation(shifted_spectrum, shifted_template, text="after shifting")
        lag = calculate_doppler_from_shift(ang_resolution * index_lag)
        print(f"lag in meters is : {lag}")

    else:
        # resample
        pass


 
    return lag, correlation






if __name__ == "__main__":
    from shifts import make_shifted_data
    from data import test_data, test_data_ideal
    a1, f1 = test_data() 
    a2, f2 = test_data()
    set_1 = [a1, f1]
    set_2 = [a2, f2]
    a1, f1, a2, f2 = make_shifted_data(set_1, set_2, velocity)
    lag, corr = calculate_correlation(set_1, set_2, "scipy")
    print(lag)
