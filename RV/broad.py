import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from resolution import increese_resolution
from specutils.fitting import continuum
from astropy.modeling.polynomial import Chebyshev1D
import PyAstronomy.pyasl as pyasl
from scipy.interpolate import interp1d
from scipy.signal import correlate


#    spectrum = np.genfromtxt("/home/lambda/code/synth_spectrum/iras2020.txt")
# template = np.genfromtxt("/home/lambda/code/synth_spectrum/iras2020.txt")
spectrum_raw = np.genfromtxt("/home/gamma/postagb.spec", usecols=(0, 1))
template_raw = np.genfromtxt("/home/gamma/postagb.spec", usecols=(0, 1))
s_AA, s_flux = increese_resolution([spectrum_raw[:, 0], spectrum_raw[:, 1]], 10)
t_AA, t_flux = increese_resolution([template_raw[:, 0], template_raw[:, 1]], 10)

lower_limit = 5000
upper_limit = 7500

mask = (s_AA >= lower_limit) & (s_AA <= upper_limit)
s_AA = s_AA[mask]
s_flux = s_flux[mask]
mask = (t_AA >= lower_limit) & (t_AA <= upper_limit)
t_AA = t_AA[mask]
t_flux = t_flux[mask]

spectrum = np.column_stack((s_AA, s_flux))
template = np.column_stack((t_AA, t_flux))

_, spectrum[:, 0] = pyasl.dopplerShift(
    spectrum[:, 0], spectrum[:, 1], 50, edgeHandling="firstlast"
)
interp_func = interp1d(
    template[:, 0], template[:, 1], kind="linear", fill_value="extrapolate"
)

intensity2_aligned = interp_func(spectrum[:, 0])
combined_intensity = spectrum[:, 1] + intensity2_aligned
two_sys = np.copy(spectrum)
two_sys[:, 0] = spectrum[:, 0]
two_sys[:, 1] = combined_intensity

flux_unit = u.Unit("erg s^-1 cm^-2 AA^-1")
unc = np.array([10e-20 for x in range(len(two_sys[:, 1]))]) * flux_unit

templateS = Spectrum1D(
    spectral_axis=template[:, 0] * u.AA, flux=template[:, 1] * flux_unit
)
observed = Spectrum1D(
    spectral_axis=two_sys[:, 0] * u.AA,
    flux=two_sys[:, 1] * flux_unit,
    uncertainty=StdDevUncertainty(unc),
)
continuum_model = continuum.fit_generic_continuum(observed, model=Chebyshev1D(7))
p_obs = observed - continuum_model(observed.wavelength)
continuum_model = continuum.fit_generic_continuum(templateS, model=Chebyshev1D(7))
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
    velocities = np.arange(-len(cross_corr) // 2, len(cross_corr) // 2) * delta_lambda

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
