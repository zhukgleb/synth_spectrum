import numpy as np
import matplotlib.pyplot as plt
from data import extract_data
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

path = "/home/alpha/git/synth_spectrum/data/Test_spectrum.syn"
ang, flux, _ = extract_data(path)


# Spectral interval
start_ang = 6000
end_ang = 6100
index_start = np.where(ang==start_ang)[0][0]
index_end = np.where(ang==end_ang)[0][0]

ang = ang[index_start:index_end]
flux = flux[index_start:index_end]


# Interpolation part
ang_inter_norm = np.linspace(np.min(ang), np.max(ang), len(ang) * 10)

y_cubicBC_norm = CubicSpline(ang, flux, bc_type="natural")
y_akima_norm = Akima1DInterpolator(ang, flux)
y_cubicMT_norm = PchipInterpolator(ang, flux)

border_1 = np.linspace(0.1, 0.1, len(ang))
border_2 = np.linspace(1, 1, len(ang))
peaks, _ = find_peaks(1/flux, prominence=0.1)
plt.plot(ang[peaks], flux[peaks], "x")
plt.plot(ang, flux, "--")
plt.show()
