from scipy.interpolate import interp1d
import numpy as np


def increese_resolution(spectrum_data, coeff: int, method: str = "quad"):
    AA_original = spectrum_data[0]
    flux_original = spectrum_data[1]
    if type(spectrum_data) is np.ndarray:
        AA_original = spectrum_data[:, 0]
        flux_original = spectrum_data[:, 1]

    AA = np.linspace(np.min(AA_original), np.max(AA_original), len(AA_original) * coeff)
    if method == "quad":
        interp_flux = interp1d(AA_original, flux_original, kind="quadratic")
    else:
        pass

    flux = interp_flux(AA)
    return AA, flux


def interpolate_in_knots(spectrum, knots):
    if type(spectrum) is np.ndarray:
        aa = spectrum[:, 0]
        flux = spectrum[:, 1]
    elif type(spectrum) is list:
        aa = spectrum[0]
        flux = spectrum[1]
    else:
        raise
    interp_flux = interp1d(aa, flux, kind="quadratic")
    flux = interp_flux(knots)

    return knots, flux


if __name__ == "__main__":
    spec = np.genfromtxt("/Users/beta/synth_spectrum/iras07430.txt")
    knots = np.arange(spec[:, 0][0], spec[:, 0][-1], 0.005)
    aa, flux = interpolate_in_knots(spec, knots)
    np.savetxt("iras07430knots.txt", np.column_stack((aa, flux)))
