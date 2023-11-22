from data import extract_data
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.analysis import correlation
from corr import calculate_correlation



a1, f1 = extract_data("data/model1_shift350.data", text=True)
a_template, f_template = extract_data("data/model1_noshift.data", text=True)

aa_start = 2000.0
aa_end = 9000.0

obs_crop = np.where((a1 >= aa_start) & (a1 <= aa_end))
template_crop = np.where((a_template >= aa_start) & (a_template <= aa_end))

a1 = a1[obs_crop]
f1 = f1[obs_crop]
a_template = a_template[template_crop]
f_template = f_template[template_crop]


flux_unit = u.Unit('erg s^-1 cm^-2 AA^-1')
unc = [0.0001 for x in range(len(f1))]

plt.plot(a1, f1)
plt.plot(a_template, f_template)
plt.show()

template = Spectrum1D(spectral_axis=a_template*u.AA, flux=f_template*flux_unit)
observed = Spectrum1D(spectral_axis=a1*u.AA, flux=f1*flux_unit, uncertainty=StdDevUncertainty(unc))

corr, lag = correlation.template_correlate(observed, template, lag_units=u.one)

plt.plot(lag, corr)
plt.xlim([-0.1,0.1])
plt.xlabel("Redshift")
plt.ylabel("Correlation Signal")
plt.show()

z_peak = lag[np.where(corr==np.max(corr))][0]
calculate_velocity = z_peak * 299792458

print(z_peak)
print(f"calculate speed is {calculate_velocity}")


n = 8 # points to the left or right of correlation maximum
index_peak = np.where(corr == np.amax(corr))[0][0]
peak_lags = lag[index_peak-n:index_peak+n+1].value
peak_vals = corr[index_peak-n:index_peak+n+1].value
p = np.polyfit(peak_lags, peak_vals, deg=2)
roots = np.roots(p)
v_fit = np.mean(roots) * u.km/u.s # maximum lies at mid point between roots
# z = v_fit / const.c.to('km/s')
z = 0
print("Parabolic fit with maximum at: ", v_fit)
print("Redshift from parabolic fit: ", z)
