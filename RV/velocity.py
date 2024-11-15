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
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev**2))


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

    if save:
        np.savetxt(
            "results/spectrum.txt", np.column_stack((spectrum_ang, spectrum_flux))
        )
        np.savetxt(
            "results/template.txt", np.column_stack((template_ang, template_flux))
        )

    flux_unit = u.Unit("erg s^-1 cm^-2 AA^-1")
    unc = np.array([10e-20 for x in range(len(spectrum_flux))]) * flux_unit

    speed_arr = []  # for speed calculate in multiply approach

    template = Spectrum1D(
        spectral_axis=template_ang * u.AA, flux=template_flux * flux_unit
    )
    observed = Spectrum1D(
        spectral_axis=spectrum_ang * u.AA,
        flux=spectrum_flux * flux_unit,
        uncertainty=StdDevUncertainty(unc),
    )

    if raw:
        continuum_model = continuum.fit_generic_continuum(
            observed, model=Chebyshev1D(6)
        )
        p_obs = observed - continuum_model(observed.wavelength)
        continuum_model = continuum.fit_generic_continuum(
            template, model=Chebyshev1D(3)
        )
        p_template = template - continuum_model(template.wavelength)
    else:
        p_obs = observed
        p_template = template

    # Needs to gentlee setting
    fc = 0.25  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.49  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).

    N = int(np.ceil((4 / b)))
    if not N % 2:  # N must be odd
        N += 1
    n = np.arange(N)

    filt = np.sinc(2 * fc * (n - (N - 1) / 2))
    w = (
        0.42
        - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    )
    filt *= w
    filt /= np.sum(filt)
    p_obs_smoothed = convolution_smooth(observed, filt)

    # Correlation part
    sigma1 = np.sqrt(1 / len(spectrum_ang) * np.sum(spectrum_flux**2))
    sigma2 = np.sqrt(1 / len(template_ang) * np.sum(template_flux**2))
    corr, lag = correlation.template_correlate(
        p_obs_smoothed, p_template, lag_units=lag_unit, method="fft"
    )
    corr = corr / (sigma1 * sigma2 * len(corr))
    corr = corr.value
    lag = np.array(lag.value * 299792458)
    z_peak = lag[np.where(corr == np.max(corr))][0]
    calculate_velocity = z_peak
    speed_arr.append(calculate_velocity)

    if plot:
        plt.plot(lag, corr, linewidth=1)
        plt.ylabel("Correlation Signal")
        plt.show()

    sigma_o_g = 0

    if do_fit:
        sigma_o_g = gauss_corr_fit([lag, corr], calculate_velocity, 0, 0)

    if save:
        np.savetxt("results/correlation.txt", np.column_stack((lag, corr)))

    del corr
    del lag

    corr_template, lag_template = correlation.template_correlate(
        p_template, p_template, lag_units=lag_unit, method="fft"
    )
    lag_template = lag_template.value * 299792458

    corr_template = corr_template / (sigma1 * sigma2 * len(corr_template))
    corr_template = corr_template.value

    if plot:
        plt.plot(lag_template * 299792458, corr_template)
        plt.ylabel("Correlation Signal")
        plt.show()

    sigma_t_g = 0
    if save:
        np.savetxt(
            "results/autocorrelation.txt",
            np.column_stack((lag_template, corr_template)),
        )

    if do_fit:
        sigma_t_g = gauss_corr_fit([lag_template, corr_template], 0, 1, 0.91)

    del lag_template
    del corr_template

    # Error count part
    sigma_t = np.std(template_flux)
    sigma_g = np.std(spectrum_flux)
    Rt = np.sqrt(np.mean(template_flux**2))
    Rg = np.sqrt(np.mean(spectrum_flux**2))
    sigma = (1 / len(template_flux)) * (sigma_t**2 / Rt**2 + sigma_g**2 / Rg**2) ** 0.5

    z_err = (sigma_o_g**2 - sigma_t_g**2) ** 0.5

    print(
        f"Direct calculated velocity is: {calculate_velocity} m/s, sigma gauss: {z_err} m/s, sigma: {sigma*299792458} m/s"
    )

    return calculate_velocity, z_peak, z_err, sigma * 299792458


if __name__ == "__main__":
    import PyAstronomy.pyasl as pyasl
    from scipy.interpolate import interp1d

    #    spectrum = np.genfromtxt("/home/lambda/code/synth_spectrum/iras2020.txt")
    # template = np.genfromtxt("/home/lambda/code/synth_spectrum/iras2020.txt")
    spectrum_raw = np.genfromtxt("/home/lambda/postagb.spec", usecols=(0, 1))
    template_raw = np.genfromtxt("/home/lambda/postagb.spec", usecols=(0, 1))
    s_AA, s_flux = increese_resolution([spectrum_raw[:, 0], spectrum_raw[:, 1]], 40)
    t_AA, t_flux = increese_resolution([template_raw[:, 0], template_raw[:, 1]], 40)

    lower_limit = 5000
    upper_limit = 5500

    mask = (s_AA >= lower_limit) & (s_AA <= upper_limit)
    s_AA = s_AA[mask]
    s_flux = s_flux[mask]
    mask = (t_AA >= lower_limit) & (t_AA <= upper_limit)
    t_AA = t_AA[mask]
    t_flux = t_flux[mask]

    spectrum = np.column_stack((s_AA, s_flux))
    template = np.column_stack((t_AA, t_flux))

    _, spectrum[:, 0] = pyasl.dopplerShift(
        spectrum[:, 0], spectrum[:, 1], 160, edgeHandling="firstlast"
    )
    interp_func = interp1d(
        template[:, 0], template[:, 1], kind="quadratic", fill_value="extrapolate"
    )

    intensity2_aligned = interp_func(spectrum[:, 0])
    combined_intensity = spectrum[:, 1] + intensity2_aligned
    two_sys = np.copy(spectrum)
    two_sys[:, 0] = spectrum[:, 0]
    two_sys[:, 1] = combined_intensity

    # mask = (two_sys[:, 0] >= lower_limit) & (two_sys[:, 0] <= upper_limit)
    # two_sys_w_crp = two_sys[:, 0][mask]
    # two_sys_f_crp = two_sys[:, 1][mask]
    # two_sys = np.column_stack((two_sys[:, 0], two_sys[:, 1]))
    flux_unit = u.Unit("erg s^-1 cm^-2 AA^-1")
    unc = np.array([10e-20 for x in range(len(two_sys[:, 1]))]) * flux_unit

    #    two_sys_int_AA, two_sys_int_flux = increese_resolution(
    #    [two_sys[:, 0], two_sys[:, 1]], 10
    # )
    # template_int_AA, template_int_flux = increese_resolution(
    #    [template[:, 0], template[:, 1]], 10
    # )
    # template = np.column_stack((template_int_AA, template_int_flux))
    # two_sys = np.column_stack((two_sys_int_AA, two_sys_int_flux))

    templateS = Spectrum1D(
        spectral_axis=template[:, 0] * u.AA, flux=template[:, 1] * flux_unit
    )
    observed = Spectrum1D(
        spectral_axis=two_sys[:, 0] * u.AA,
        flux=two_sys[:, 1] * flux_unit,
        uncertainty=StdDevUncertainty(unc),
    )
    continuum_model = continuum.fit_generic_continuum(observed, model=Chebyshev1D(15))
    p_obs = observed - continuum_model(observed.wavelength)
    continuum_model = continuum.fit_generic_continuum(templateS, model=Chebyshev1D(15))
    p_template = templateS - continuum_model(templateS.wavelength)

    template[:, 0] = p_template.wavelength
    template[:, 1] = p_template.flux

    two_sys[:, 0] = p_obs.wavelength
    two_sys[:, 1] = p_obs.flux

    """
    cv, zp, ze, sigma = find_velocity(
        [spectrum[:, 0], spectrum[:, 1]],
        [template[:, 0], template[:, 1]],
        [5200, 5300],
        100,
    )
    """
    from scipy.signal import correlate

    def broadening_function(object_spectrum, template_spectrum):
        # Развертываем данные
        obj_wavelengths, obj_intensities = (
            np.log(object_spectrum[:, 0]),
            object_spectrum[:, 1],
        )
        template_wavelengths, template_intensities = (
            np.log(template_spectrum[:, 0]),
            template_spectrum[:, 1],
        )

        # Интерполяция на общую сетку (например, сетку объекта)
        interp_template = interp1d(
            template_wavelengths,
            template_intensities,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        template_on_obj_grid = interp_template(obj_wavelengths)

        # Кросс-корреляция
        cross_corr = correlate(obj_intensities, template_on_obj_grid, mode="same")

        # Нормировка кросс-корреляции (опционально)
        cross_corr /= np.max(cross_corr)

        # Шаг между точками для соответствия шкале скоростей
        delta_lambda = obj_wavelengths[1] - obj_wavelengths[0]
        velocities = (
            np.arange(-len(cross_corr) // 2, len(cross_corr) // 2) * delta_lambda
        )

        return velocities, cross_corr

    import numpy as np
    import matplotlib.pyplot as plt

    # Функция для пересчета длин волн в скорости
    def calculate_velocities(wavelengths, delta_lambda):
        c = 299792  # скорость света в км/с
        lambda_0 = np.mean(
            wavelengths
        )  # средняя длина волны, можно изменить при необходимости
        velocities = delta_lambda / lambda_0 * c
        return velocities

    def plot_broadening_function(wavelengths, bf_profile):
        # Расчет шага по длине волны
        delta_lambda = wavelengths[1] - wavelengths[0]

        # Пересчет в скорости
        velocities = calculate_velocities(
            wavelengths,
            delta_lambda * np.arange(-len(bf_profile) // 2, len(bf_profile) // 2),
        )

        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(velocities, bf_profile, color="blue", label="Broadening Function")
        plt.xlabel("Velocity (km/s)")
        plt.ylabel("BF Intensity")
        plt.title("Broadening Function Profile")
        plt.legend()
        plt.grid(True)
        plt.show()

    v, bf = broadening_function(two_sys, template)
    plot_broadening_function(template[:, 0], bf)
