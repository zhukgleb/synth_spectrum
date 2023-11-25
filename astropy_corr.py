from data import extract_data
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.analysis import correlation
from resolution import increese_resolution
import PyAstronomy.pyasl as pyasl
import astropy.constants as const

def find_speed(spectrum: list, template: list, inter: list, mult: int):
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
    unc = [0.00000001 for x in range(len(spectrum_flux))]

    speed_arr = []  # for speed calculate in multiply approach

    template = Spectrum1D(spectral_axis=template_ang*u.AA, flux=template_flux*flux_unit)
    observed = Spectrum1D(spectral_axis=spectrum_ang*u.AA, flux=spectrum_flux*flux_unit, uncertainty=StdDevUncertainty(unc))
    corr, lag = correlation.template_correlate(observed, template, lag_units=u.one)

#        plt.plot(lag*299792458, corr)
#        plt.xlabel("Correlation speed, m/s")
#        plt.ylabel("Correlation Signal")
#        plt.show()

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
    v_fit = np.mean(roots) * u.m/u.s # maximum lies at mid point between roots
    z = v_fit * 299792458 
#     print("Parabolic fit with maximum at: ", v_fit)
    print("Redshift from parabolic fit: ", z)
#        plt.scatter(peak_lags * 299792458, peak_vals, label='data')
#        plt.plot(peak_lags * 299792458, np.polyval(p, peak_lags), linewidth=0.5, label='fit')
 #       plt.xlabel(lag.unit)
 #       plt.legend()
 #       plt.title('Fit to correlation peak')
 #       plt.show()
    return calculate_velocity, z

        
#        _, specturm_ang = pyasl.dopplerShift(spectrum_ang, spectrum_flux, -calculate_velocity/1000, edgeHandling="firstlast")
    print(f"total velocity sum is: {sum(speed_arr)}")


if __name__ == "__main__":
#     s1 = extract_data("data/Test_spectrum.syn", text=True)
    s2 = extract_data("data/NES_model_110000.rgs", text=True)
    s3 = extract_data("data/NES_model_60000.rgs", text=True)
    s4 = extract_data("data/NES_model_40000.rgs", text=True)
    s5 = extract_data("data/NES_model_15000.rgs", text=True)
    a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

    spectrum_arr = [s2, s3, s4, s5]
    spectrum_names = ["s2", "s3", "s4", "s5"]
    #
    # 1 is meter
    for i in range(len(spectrum_arr)):
        with open("calc/" + spectrum_names[i] + ".txt", 'w') as f:
            for j in range(1, 1000):
                print(f"speed is {j} meters")
                _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], j/1000, edgeHandling="firstlast")
                mult = 10
                cv, z = find_speed(spectrum_arr[i], [a_template, f_template], [5000, 5100], mult)
                string = str(j) + "\t"  + str(cv) + "\t" + str(z) + "\n"
                f.write(string)
                _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], -j/1000, edgeHandling="firstlast")
