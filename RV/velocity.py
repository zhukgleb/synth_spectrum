import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.analysis import correlation
from resolution import increese_resolution
from fit import gauss_corr_fit
from specutils.fitting import continuum
from astropy.modeling.polynomial import Chebyshev1D
from specutils.manipulation import gaussian_smooth, convolution_smooth




def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

# @profile

# TODO:
# Rewrite a part, where corrs calculate to:
# template->template correlation, error estimation -> delete garbage.
# observed->template correlation, error estimation -> delete garbage.
# Thinking about re-calculation template-template correlation, with multiply
# instances
# and cut then multiply dots

def find_velocity(spectrum: list, template: list, inter: list, mult: int):
    do_fit = False
    plot = True
    raw = True
    save = False
    lag_unit = u.one
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


    if plot:
        plt.plot(spectrum_ang, spectrum_flux, linewidth=1, label='obs')
        plt.plot(template_ang, template_flux, linewidth=0.5, color='r', label='template')
        plt.show()

    if save:
        np.savetxt("results/spectrum.txt", np.column_stack((spectrum_ang, spectrum_flux)))
        np.savetxt("results/template.txt", np.column_stack((template_ang, template_flux)))

    
    flux_unit = u.Unit('erg s^-1 cm^-2 AA^-1')
    unc = np.array([10e-20 for x in range(len(spectrum_flux))]) * flux_unit

    speed_arr = []  # for speed calculate in multiply approach

    template = Spectrum1D(spectral_axis=template_ang*u.AA, 
                          flux=template_flux*flux_unit)
    observed = Spectrum1D(spectral_axis=spectrum_ang*u.AA,
                          flux=spectrum_flux*flux_unit,
                          uncertainty=StdDevUncertainty(unc))
    
    if raw:
        continuum_model = continuum.fit_generic_continuum(observed, model=Chebyshev1D(15))
        p_obs = observed - continuum_model(observed.wavelength)
        continuum_model = continuum.fit_generic_continuum(template, model=Chebyshev1D(15)) 
        p_template = template - continuum_model(template.wavelength)
    else:
        p_obs = observed
        p_template = template

    
    # Needs to gentlee setting

    fc = 0.25  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.49   # Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    N = int(np.ceil((4 / b)))
    if not N % 2:  # N must be odd
        N += 1
    n = np.arange(N)
     
    filt = np.sinc(2 * fc * (n - (N - 1) / 2))
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))
    filt *= w
    filt /= np.sum(filt)
    p_obs_smoothed = convolution_smooth(observed, filt)

    if plot:
        plt.plot(p_obs.wavelength, p_obs.flux, linewidth=1, label='obs')
        plt.plot(p_template.wavelength, p_template.flux, linewidth=0.5, color='r', label='template')
        plt.show()



    # Correlation part

    sigma1 = np.sqrt(1/len(spectrum_ang) * np.sum(spectrum_flux**2))
    sigma2 = np.sqrt(1/len(template_ang) * np.sum(template_flux**2))
    corr, lag = correlation.template_correlate(p_obs_smoothed, p_template, lag_units=lag_unit, method="fft")
    corr = corr / (sigma1 * sigma2 * len(corr))
    corr = corr.value
    lag = np.array(lag.value * 299792458)
 
    z_peak = lag[np.where(corr==np.max(corr))][0]

    if lag_unit == u.one:  # if quant of lags is u.one! re-write later
        calculate_velocity = z_peak
    else:
        calculate_velocity = z_peak
    speed_arr.append(calculate_velocity)
 
   
    if plot:
        plt.plot(lag, corr, linewidth=1)
        plt.xlim((-10*1000*1000, 10*1000*1000))
        plt.ylabel("Correlation Signal")
        plt.show()
   
    sigma_o_g = 0

    if do_fit:
        sigma_o_g = gauss_corr_fit([lag, corr], calculate_velocity, 0, 0)


    if save:
        np.savetxt("results/correlation.txt", np.column_stack((lag, corr)))
  
    del corr
    del lag

    corr_template, lag_template = correlation.template_correlate(p_template,
                                                                 p_template,
                                                                 lag_units=lag_unit,
                                                                 method="fft")
    lag_template = lag_template.value * 299792458

    corr_template = corr_template / (sigma1 * sigma2 * len(corr_template))
    corr_template = corr_template.value

    if plot:
        plt.plot(lag_template * 299792458, corr_template)
        plt.ylabel("Correlation Signal")
        plt.show()

    sigma_t_g = 0
    if save:
        np.savetxt("results/autocorrelation.txt", np.column_stack((lag_template, corr_template)))
 
    if do_fit:
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
