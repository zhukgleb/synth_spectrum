import numpy as np
import matplotlib.pyplot as plt

from specutils import analysis

from specutils import Spectrum1D
from astropy import units as u
from astropy.nddata import StdDevUncertainty 
from scipy.signal import correlate
from scipy.signal import correlation_lags
from shifts import calculate_doppler_from_shift

from resolution import increese_resolution




start_ang = 2500.0
end_ang = 8500.0
velocity = 350


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

    lag_corr_arr = np.column_stack((lags, correlation_result))
    sortd_lag_corr_arr = lag_corr_arr[lag_corr_arr[:, 1].argsort()]

#    plt.title("Lag-correlation")
#    plt.plot(sortd_lag_corr_arr[:, 0], sortd_lag_corr_arr[:, 1])
#    plt.show()

    print(f"Best lag: {lag}")
    if lag < 0:
        series_2 = series_2[-lag:]
    else:
        series_1 = series_1[lag:]
    return series_1, series_2, lag, sortd_lag_corr_arr


def calculate_correlation(spectrum_np, template_spectrum_np, method: str="astropy",
                          dots_mult: int=100):
    lag = 0
    corr = []
    
    a_spectrum = spectrum_np[0]
    f_spectrum = spectrum_np[1]
    a_template = template_spectrum_np[0]
    f_template = template_spectrum_np[1]

    obs_crop = np.where((a_spectrum >= start_ang) & (a_spectrum <= end_ang))
    template_crop = np.where((a_template >= start_ang) & (a_template <= end_ang))

    a_spectrum = a_spectrum[obs_crop]
    f_spectrum = f_spectrum[obs_crop]
    a_template = a_template[template_crop]
    f_template = f_template[template_crop]



    # Increese resolution by polynomial interpolation 
    a_spectrum, f_spectrum = increese_resolution([a_spectrum, f_spectrum], dots_mult)
    a_template, f_template = increese_resolution([a_template, f_template], dots_mult)


    if method == "astropy":
        flux_unit = u.Unit('erg s^-1 cm^-2 AA^-1')
        unc = [0.0001 for x in range(len(f1))]

        plt.plot(a_spectrum, f_spectrum) 
        plt.plot(a_template, f_template)
        plt.show()

        template = Spectrum1D(spectral_axis=a_template*u.AA, flux=f_template*flux_unit)
        observed = Spectrum1D(spectral_axis=a1*u.AA, flux=f1*flux_unit, uncertainty=StdDevUncertainty(unc))

        corr, lag = analysis.correlation.template_correlate(observed, template, lag_units=u.one)

        plt.plot(lag, corr)
        plt.xlim([-0.1,0.1])
        plt.xlabel("Redshift")
        plt.ylabel("Correlation Signal")
        plt.show()

        z_peak = lag[np.where(corr==np.max(corr))][0]
        calculate_velocity = z_peak * 299792458

        print(f"calculate speed is {calculate_velocity}")



    elif method == "scipy":
        spectrum_flux = f_spectrum
        template_flux = f_template
        
        # plot_correlation(f_spectrum, f_template, text="before")
        shifted_spectrum, shifted_template, index_lag, lags = shift_for_maximum_correlation(spectrum_flux, template_flux)
        # plot_correlation(shifted_spectrum, shifted_template, text="after shifting")
        ang_resolution = a_spectrum[1] - a_spectrum[0] 
        ang_lag = ang_resolution * index_lag
        print(f"AA lag {ang_lag}")
        lag = calculate_doppler_from_shift(ang_lag)
        print(f"lag in meters is : {lag}")
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(f_spectrum)
        ax1.plot(f_template)
        ax2.plot(shifted_spectrum)
        ax2.plot(shifted_template)
        plt.show()
        
#        for i in range(20):
#            print(f"Doppler shifts {i}: {calculate_doppler_from_shift(lags[i] * ang_resolution)}")
#        print(calculate_doppler_from_shift(sortd_lag_corr_arr[i]))
        


    else:
        # resample
        pass
 
    return lag, corr


if __name__ == "__main__":
    from data import extract_data
    from shifts import make_doppler_shift
    a1, f1 = extract_data("data/model1_shift350_rightversion.data", text=True)
#    a1, f1 = make_doppler_shift(a1, f1)
    a_template, f_template = extract_data("data/model1_noshift.data", text=True)
    calculate_correlation([a1, f1], [a_template, f_template], method="astropy")
