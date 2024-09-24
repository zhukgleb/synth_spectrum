from data import extract_data
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl
import numpy as np
import matplotlib.pyplot as plt


def make_good():
    # Define a n system of line
    component_a = extract_data("data/NES_model_80000.rgs", text=True)
    component_b = extract_data("data/NES_model_80000.rgs", text=True)

    a_template_a, f_template_a = extract_data("data/NES_model_80000.rgs", text=True)
    a_template_b, f_template_b = extract_data("data/NES_model_80000.rgs", text=True)


    spectrum_arr = [component_a, component_b]
    template_arr = [[a_template_a, f_template_a], [a_template_b, f_template_b]]

    v_1 = 200 # in meters
    v_2 = 1000 * 1000
    dots = 40

    ang_a = spectrum_arr[0][0]
    flux_a = spectrum_arr[0][1]
    ang_b = spectrum_arr[1][0]
    flux_b = spectrum_arr[1][1]

    _, ang_a = pyasl.dopplerShift(ang_a, flux_a, v_1 / 1000, edgeHandling="firstlast")
    _, ang_b = pyasl.dopplerShift(ang_b, flux_b, v_2 / 1000, edgeHandling="firstlast")

#    noise_a = np.random.normal(loc=0, scale=1/100, size=len(flux_a))
    # com_ang = np.concatenate((ang_a, ang_b)) 
    # com_flux = np.concatenate((noise_spectrum_a, noise_spectrum_b))

#    com_ang = []
#    com_flux = []
#    com_ang = np.append(ang_a, ang_b)
#    com_flux = np.append(flux_a, flux_b)

#    spec_arr = np.column_stack((com_ang, com_flux))
#    spec_arr = spec_arr[spec_arr[:, 0].argsort()]
    from specutils import Spectrum1D 
    import astropy.units as u
    from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
    flux_unit = u.Unit("erg s^-1 cm^-2 AA^-1")
    spec1 = Spectrum1D(spectral_axis=ang_a*u.AA,
                          flux=flux_a*flux_unit)
    spec2 = Spectrum1D(spectral_axis=ang_b*u.AA,
                          flux=flux_b*flux_unit)
    new_spectral_axis = np.concatenate([spec1.spectral_axis.value, spec2.spectral_axis.to_value(spec1.spectral_axis.unit)]) * spec1.spectral_axis.unit
    resampler = LinearInterpolatedResampler()

    new_spec1 = resampler(spec1, new_spectral_axis)
    new_spec2 = resampler(spec2, new_spectral_axis)
    final_spec = new_spec1 + new_spec2
    plt.plot(final_spec.spectral_axis, final_spec.flux)
    plt.show()

#    from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
#    spline = SplineInterpolatedResampler()
#    new_disp_grid = np.arange(min(spec_arr[:, 0]), max(spec_arr[:, 1]), 1) * u.AA
#    new_spec_sp = spline(spectrum, new_disp_grid)  

#    com_ang = spectrum.wavelength
#    com_flux = spectrum.flux

#    cv, z, z_err, s = find_velocity([com_ang, com_flux], 
#                                    [a_template_a, f_template_a],
#                                    [7000, 7700], dots)

if __name__ == "__main__":
    make_good()
