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




start_ang = 4500
end_ang = 5500
# start_ang = 5000
# end_ang = 6500


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
    series_1_log = series_1
    series_2_log = series_2
    correlation_result = correlate(series_1_log, series_2_log, mode="full", method="direct")
    lags = correlation_lags(series_1.size, series_2.size, mode="full")
    lag = lags[np.argmax(correlation_result)]

#    delta_log_wave = np.log10(series_1[1]) - np.log10(series_1[0])
#    deltas = (np.array(range(len(correlation_result))) - len(correlation_result)/2 + 0.5) * delta_log_wave
#    lags_10 = np.power(10., deltas) - 1.

#    from astropy.units import Quantity
#    lags_10 = Quantity(lags_10, u.dimensionless_unscaled)
   
    print(f"Best lag: {lag}")
    if lag < 0:
        series_2 = series_2[-lag:]
    else:
        series_1 = series_1[lag:]
    return series_1, series_2, lag


def calculate_correlation(spectrum_np, template_spectrum_np, method: str="astropy",
                          dots_mult: int=50):
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

    # resample 
    flux_unit = u.Unit('erg s^-1 cm^-2 AA^-1')
    wblue = start_ang

    wred = end_ang
    w0 = np.log10(wblue)
    w1 = np.log10(wred)
    
    ds = np.log10(a_spectrum[1:]) - np.log10(a_spectrum[:-1])
    dw = ds[np.argmin(ds)]
    a = np.mean(ds)
    print(a)
    print(dw)

    nsamples = int((w1 - w0) / dw)
    log_wave_array = np.ones(nsamples) * w0 

    for i in range(nsamples):
        log_wave_array[i] += dw * i

    wave_array = np.power(10., log_wave_array) * u.AA
    unc = [1e-6 for x in range(len(f_spectrum))]
    spectrum = Spectrum1D(spectral_axis=a_spectrum*u.AA, flux=f_spectrum*flux_unit, uncertainty=StdDevUncertainty(unc))
    template = Spectrum1D(spectral_axis=a_template*u.AA, flux=f_template*flux_unit)
    import specutils
    resampler=specutils.manipulation.SplineInterpolatedResampler()
    resampled_spectrum = resampler(spectrum, wave_array)
    resampled_template = resampler(template, wave_array)
    clean_spectrum_flux = np.nan_to_num(resampled_spectrum.flux.value) * resampled_spectrum.flux.unit
    clean_template_flux = np.nan_to_num(resampled_template.flux.value) * resampled_template.flux.unit


    clean_spectrum = Spectrum1D(spectral_axis=resampled_spectrum.spectral_axis,
                        flux=clean_spectrum_flux,
                        uncertainty=resampled_spectrum.uncertainty,
                        velocity_convention='optical',
                        rest_value=spectrum.rest_value)
    clean_template = Spectrum1D(spectral_axis=resampled_template.spectral_axis,
                        flux=clean_template_flux,
                        uncertainty=resampled_template.uncertainty,
                        velocity_convention='optical',
                        rest_value=template.rest_value)
    
    
    f_spectrum = clean_spectrum.flux.value
    f_template = clean_template.flux.value
    a_spectrum = clean_spectrum.spectral_axis.value
    a_template = clean_template.spectral_axis.value


    corr = correlate(clean_spectrum_flux, clean_template_flux, method="direct")
    lags = correlation_lags(clean_spectrum_flux.size, clean_template_flux.size, mode="full")
    lag = lags[np.argmax(corr)]

    wave_l = clean_spectrum.spectral_axis.value 
    delta_log_wave = np.log10(wave_l[1]) - np.log10(wave_l[0])
    deltas = (np.array(range(len(corr))) - len(corr)/2 + 0.5) * delta_log_wave
    lags_as = np.power(10., deltas) - 1.
    temp_val = lags_as[np.argmax(corr)] * 299792458
    print(temp_val)





    lag_corr_arr = np.column_stack((lags, corr))
    sortd_lag_corr_arr = lag_corr_arr[lag_corr_arr[:, 1].argsort()]

    plt.title("Lag-correlation")
    plt.plot(sortd_lag_corr_arr[:, 0], sortd_lag_corr_arr[:, 1])
    plt.show()


    print(f"Best lag: {lag}")
    if lag < 0:
        clean_template_flux = clean_template_flux[-lag:]
    else:
        clean_spectrum_flux = clean_spectrum_flux[lag:]

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(a_spectrum, f_spectrum)
    ax1.plot(a_template, f_template)
    ax2.plot(clean_spectrum_flux)
    ax2.plot(clean_template_flux)
    plt.show()

    if method == "astropy":
        flux_unit = u.Unit('erg s^-1 cm^-2 AA^-1')
        unc = [0.0001 for x in range(len(f1))]
        template = Spectrum1D(spectral_axis=a_template*u.AA, flux=f_template*flux_unit)
        observed = Spectrum1D(spectral_axis=a1*u.AA, flux=f1*flux_unit, uncertainty=StdDevUncertainty(unc))
        corr, lag = analysis.correlation.template_correlate(observed, template, lag_units=u.one)
        z_peak = lag[np.where(corr==np.max(corr))][0]
        calculate_velocity = z_peak * 299792458
        print(f"velocity is {calculate_velocity}")


    elif method == "scipy":
        return lag, corr

    return lag, corr


if __name__ == "__main__":
    from data import extract_data, test_data, test_data_ideal
    from shifts import make_doppler_shift
#     a1, f1 = extract_data("data/model1_shift350.data", text=True)
#    a1, f1 = make_doppler_shift(a1, f1, 350 * 1000)
#    a1, f1 = test_data()
#    a_template, f_template = test_data_ideal()
#    a1, f1 = make_doppler_shift(a1, f1, 350*1000) 
#
#    a_template, f_template = extract_data("data/model1_noshift.data", text=True)

    a1, f1 = extract_data("models/Good_model.rgs", text=True)
    a_template, f_template = extract_data("models/Good_model.syn", text=True)
    a1, f1 = make_doppler_shift(a1, f1, 1000)
    calculate_correlation([a1, f1], [a_template, f_template], method="scipy")
