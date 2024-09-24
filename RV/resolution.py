from scipy.interpolate import interp1d
import numpy as np



def increese_resolution(spectrum_data, coeff, method: str="quad"):
    AA_original = spectrum_data[0]
    flux_original = spectrum_data[1]
    AA = np.linspace(np.min(AA_original), np.max(AA_original), len(AA_original) * coeff)
    if method=="quad":
        interp_flux = interp1d(AA_original, flux_original, kind="quadratic")
    else:
        pass

    flux = interp_flux(AA)
    return AA, flux
