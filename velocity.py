import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.analysis import correlation
from resolution import increese_resolution
from memory_profiler import profile
from fit import gauss_corr_fit



# @profile

# TODO:
# Rewrite a part, where corrs calculate to:
# template->template correlation, error estimation -> delete garbage.
# observed->template correlation, error estimation -> delete garbage.
# Thinking about re-calculation template-template correlation, with multiply
# instances
# and cut then multiply dots

def find_velocity(spectrum: list, template: list, inter: list, mult: int):
    plot = False
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
    unc_t = [10e-10 for x in range(len(template_flux))]

    speed_arr = []  # for speed calculate in multiply approach

    template = Spectrum1D(spectral_axis=template_ang*u.AA, 
                          flux=template_flux*flux_unit,
                          uncertainty=StdDevUncertainty(unc_t))
    observed = Spectrum1D(spectral_axis=spectrum_ang*u.AA,
                          flux=spectrum_flux*flux_unit,
                          uncertainty=StdDevUncertainty(unc))
    corr, lag = correlation.template_correlate(observed, template,
                                               lag_units=u.one, method="fft")
    corr = (corr - np.min(corr)) / (np.max(corr) - np.min(corr))  # normalize correlation

    if plot:
        plt.plot(lag*299792458, corr)
        plt.xlabel("Correlation speed, m/s")
        plt.ylabel("Correlation Signal")
        plt.show()

    z_peak = lag[np.where(corr==np.max(corr))][0]
    calculate_velocity = z_peak * 299792458
    speed_arr.append(calculate_velocity)

    sigma_o_g = gauss_corr_fit([lag, corr], calculate_velocity, 3000, 0.965)

    del corr
    del lag

    corr_template, lag_template = correlation.template_correlate(template,
                                                                 template,
                                                                 lag_units=u.one,
                                                                 method="fft")
    corr_template = (corr_template - np.min(corr_template)) / (np.max(corr_template) - np.min(corr_template)) 
 
#    n = 15 * 1000 # points to the left or right of correlation maximum
#    index_peak = np.where(corr == np.amax(corr))[0][0]
#    peak_lags = lag[index_peak-n:index_peak+n+1].value
#    peak_vals = corr[index_peak-n:index_peak+n+1].value
#    p = np.polyfit(peak_lags, peak_vals, deg=2)
#    roots = np.roots(p)
#    v_fit = np.mean(roots) # maximum lies at mid point between roots
#    z = v_fit * 299792458 
    sigma_t_g = gauss_corr_fit([lag_template, corr_template], 0, 1, 0.91)
    del lag_template
    del corr_template

#    if plot:
#        plt.scatter(peak_lags * 299792458, peak_vals, label='data')
#        plt.plot(peak_lags * 299792458, np.polyval(p, peak_lags),
#                 linewidth=0.5, label='fit')
#        plt.xlabel(lag.unit)
#        plt.legend()
#        plt.title('Fit to correlation peak')
#        plt.show()

    # Error count part
    sigma_t = np.std(template_flux)
    sigma_g = np.std(spectrum_flux)
    Rt = np.sqrt(np.mean(template_flux**2))
    Rg = np.sqrt(np.mean(spectrum_flux**2))
    sigma =  (1/len(template_flux)) * (sigma_t**2 / Rt**2 + sigma_g**2 / Rg**2) ** 0.5

    # A is cross-correlation peak intens
    # THIS PEACE OF SHIT
    # Calculate velocity error
    # Moiseev error counting
    # Optimization correlation peak with gauss function
    # For this, need to cut-off some lenght of arrray
    # TO DO: add cuts for multiply peaks fitting.
    # cut of template-template correlation:
#   if plot:
#        plt.plot(lag_template*299792458, gaussian_function(lag_template*299792458,
#                                                           *fit_params), 'r-', label='fit')
#        plt.plot(lag_template*299792458, corr_template)
#        plt.xlabel("Correlation speed, m/s")
#        plt.ylabel("Correlation Signal")
#        plt.show()


#   if plot:
#        plt.plot(lag*299792458, gaussian_function(lag*299792458, *fit_params), 'r-', label='fit')
#        plt.plot(lag*299792458, corr)
#        plt.xlabel("Correlation speed, m/s")
#        plt.ylabel("Correlation Signal")
#        plt.show()

    z_err = sigma_o_g**2 - sigma_t_g**2
    z_err = z_err**0.5 
    
    print(f"Direct calculated velocity is: {calculate_velocity} m/s, sigma gauss: {z_err} m/s, sigma: {sigma*299792458} m/s")

    return calculate_velocity, z_peak, z_err, sigma*299792458
