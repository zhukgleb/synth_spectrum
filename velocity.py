import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.analysis import correlation
from resolution import increese_resolution


def gaussian_function(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + 0.96


def find_velocity(spectrum: list, template: list, inter: list, mult: int):
    spectrum_ang = spectrum[0]
    spectrum_flux = spectrum[1]
    template_ang = template[0]
    template_flux = template[1]
    spectrum_ang, spectrum_flux = increese_resolution(spectrum, mult)
    template_ang, template_flux = increese_resolution(template, mult)

    aa_start = inter[0]
    aa_end = inter[1]

    obs_crop = np.where((spectrum_ang >= aa_start) & (spectrum_ang <= aa_end))
    template_crop = np.where((template_ang >= aa_start) & (template_ang <= aa_end))

    spectrum_ang = spectrum_ang[obs_crop]
    spectrum_flux = spectrum_flux[obs_crop]
    template_ang = template_ang[template_crop]
    template_flux = template_flux[template_crop]

    flux_unit = u.Unit('erg s^-1 cm^-2 AA^-1')
    unc = [10e-10 for x in range(len(spectrum_flux))]

    speed_arr = []  # for speed calculate in multiply approach

    template = Spectrum1D(spectral_axis=template_ang*u.AA, flux=template_flux*flux_unit)
    observed = Spectrum1D(spectral_axis=spectrum_ang*u.AA, flux=spectrum_flux*flux_unit, uncertainty=StdDevUncertainty(unc))
    corr, lag = correlation.template_correlate(observed, template, lag_units=u.one, method="fft")
    corr = (corr-np.min(corr))/(np.max(corr)-np.min(corr)) 
#    plt.plot(lag*299792458, corr)
#    plt.xlabel("Correlation speed, m/s")
#    plt.ylabel("Correlation Signal")
#    plt.show()

    z_peak = lag[np.where(corr==np.max(corr))][0]
    calculate_velocity = z_peak * 299792458
    speed_arr.append(calculate_velocity)
    print(f"calculate speed is {calculate_velocity}")
    n = 8 # points to the left or right of correlation maximum
    index_peak = np.where(corr == np.amax(corr))[0][0]
    peak_lags = lag[index_peak-n:index_peak+n+1].value
    peak_vals = corr[index_peak-n:index_peak+n+1].value
    p = np.polyfit(peak_lags, peak_vals, deg=2)
    roots = np.roots(p)
    v_fit = np.mean(roots) # maximum lies at mid point between roots
    z = v_fit * 299792458 
    print("Redshift from parabolic fit: ", z)
#    plt.scatter(peak_lags * 299792458, peak_vals, label='data')
#    plt.plot(peak_lags * 299792458, np.polyval(p, peak_lags), linewidth=0.5, label='fit')
#    plt.xlabel(lag.unit)
#    plt.legend()
#    plt.title('Fit to correlation peak')
#    plt.show()

    # Error count part
    sigma_t = np.std(template_flux)
    sigma_g = np.std(spectrum_flux)
    Rt = np.sqrt(np.mean(template_flux**2))
    Rg = np.sqrt(np.mean(spectrum_flux**2))
    sigma =  (1/len(template_flux)) * (sigma_t**2 / Rt**2 + sigma_g**2 / Rg**2) ** 0.5
    print(f"sigma: {sigma}")
    print(f"sigma in meters {sigma*299792458}")
    print("--------------------------------")
    # A is cross-correlation peak intens
    # THIS PEACE OF SHIT
    # Calculate velocity error
#    A = 1 # for now it's a one, because spectrum is normalize
#    from scipy.signal import find_peaks
#    from scipy import stats
#    from scipy.optimize import curve_fit
#    from scipy import asarray as ar,exp
#    peaks, _ = find_peaks(1 / spectrum_flux, prominence=0.1)
#    plt.plot(spectrum_ang, spectrum_flux)
#    plt.plot(spectrum_ang[peaks], spectrum_flux[peaks], "+", color='red')
#    plt.show()
#    N_lines = len(peaks)
#    mean_hwhm = 0.2  # 0.25?
#    delta_x = spectrum_ang[1] - spectrum_ang[0]
#    W_g = 0
#    for i in range(N_lines):
#        W_g+=1-spectrum_flux[peaks[i]] * 3.14 **0.5 * mean_hwhm
#    W_g = W_g * N_lines**(-1)
#    sigma_v = ((2*2**0.5)**0.5*sigma_g*mean_hwhm*delta_x)/(A*W_g*N_lines**0.5)
#    print(sigma_v * 299792458)
#    another_sigma_v = (1/sigma) * (1**2 * 3.14**0.5) / (2*mean_hwhm*delta_x)
#    print(another_sigma_v / 2299792458)

    # Moiseev error counting
    # warning! rewriting some vars
    unc = [10e-10 for x in range(len(template_flux))]
    template = Spectrum1D(spectral_axis=template_ang*u.AA, flux=template_flux*flux_unit)
    observed = Spectrum1D(spectral_axis=template_ang*u.AA, flux=template_flux*flux_unit, uncertainty=StdDevUncertainty(unc)) # not observed! another template!
    corr_template, lag_template = correlation.template_correlate(observed, template, lag_units=u.one, method="fft")
    corr_template = (corr_template-np.min(corr_template))/(np.max(corr_template)-np.min(corr_template)) 

    autocorr_crop = np.where((lag_template*299792458 >= -15*1000) & (lag_template*299792458<= 15*1000))
    corr_template = corr_template[autocorr_crop]
    lag_template = lag_template[autocorr_crop]


    from scipy.optimize import curve_fit
    initial_guess = [1, 0, 1]
    fit_params, covariance = curve_fit(gaussian_function, lag_template*299792458, corr_template, p0=initial_guess, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    fit_sigma = fit_params[2]
    print(fit_sigma)
    plt.plot(lag_template*299792458, gaussian_function(lag_template*299792458, *fit_params), 'r-', label='fit')
    plt.plot(lag_template*299792458, corr_template)
    plt.xlabel("Correlation speed, m/s")
    plt.ylabel("Correlation Signal")
    plt.show()




    corr_crop = np.where((lag*299792458 >= 5*1000) & (lag*299792458<= 40*1000))
    corr = corr[corr_crop]
    lag = lag[corr_crop]


    initial_guess = [1, 20*1000, 5000]
    fit_params, covariance = curve_fit(gaussian_function, lag*299792458, corr, p0=initial_guess, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    fit_sigma = fit_params[2]
    print(fit_sigma)
    plt.plot(lag*299792458, gaussian_function(lag*299792458, *fit_params), 'r-', label='fit')
    plt.plot(lag*299792458, corr)
    plt.xlabel("Correlation speed, m/s")
    plt.ylabel("Correlation Signal")
    plt.show()



    return calculate_velocity, z
