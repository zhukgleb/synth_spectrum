import numpy as np
import matplotlib.pyplot as plt
from data import extract_data, find_nearest, get_path2
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

ang_1, flux_1, _ = extract_data(get_path2("Test_spectrum.syn"))
ang_2, flux_2, _ = extract_data(get_path2("Test_spectrum_bad.syn"))

# Spectral interval
start_ang = 4600
end_ang = 6400

index_start_1 = np.where(ang_1==start_ang)[0][0]
index_end_1 = np.where(ang_1==end_ang)[0][0]
ang_1 = ang_1[index_start_1:index_end_1]
flux_1 = flux_1[index_start_1:index_end_1]

index_start_2 = np.where(ang_2==start_ang)[0][0]
index_end_2 = np.where(ang_2==end_ang)[0][0]
ang_2 = ang_2[index_start_2:index_end_2]
flux_2 = flux_2[index_start_2:index_end_2]


# Interpolation part
ang_inter_norm = np.linspace(np.min(ang_1), np.max(ang_1), len(ang_1) * 10)
ang_inter = np.linspace(np.min(ang_2), np.max(ang_2), len(ang_1) * 10)
y_cubicBC_1 = CubicSpline(ang_1, flux_1, bc_type="natural")
y_akima_1 = Akima1DInterpolator(ang_1, flux_1)
y_cubicMT_1 = PchipInterpolator(ang_1, flux_1)
y_cubicBC_2 = CubicSpline(ang_2, flux_2, bc_type="natural")
y_akima_2 = Akima1DInterpolator(ang_2, flux_2)
y_cubicMT_2 = PchipInterpolator(ang_2, flux_2)

# Difference part
peaks_1, _ = find_peaks(1/flux_1, prominence=0.1)
peaks_2_inter, _ = find_peaks(1/y_cubicBC_2(ang_inter), prominence=0.1)

# Plot part
# plt.plot(ang_inter[peaks_2_inter], y_cubicBC_2(ang_inter)[peaks_2_inter], ".", color="blue")
# plt.plot(ang_inter_norm, y_cubicBC_2(ang_inter_norm))
plt.plot(ang_inter, y_cubicBC_2(ang_inter), label="Interpolated dots from 1A res")
plt.xlabel("Wavelenght, Ang")
plt.ylabel("Relative flux")
plt.plot(ang_1, flux_1, "--", label='Ideal spectrum, 10E-3A res')
plt.plot(ang_1[peaks_1], flux_1[peaks_1], ".", color="red")
p1 = ang_1[peaks_1]
p2 = ang_inter[peaks_2_inter]
plt.plot(p2, y_cubicBC_2(ang_inter)[peaks_2_inter], ".", color="blue") 
plt.legend()
plt.show()
# If we have a different number of peaks

if len(p1) > len(p2):
    delta_center_arr = [find_nearest(p1, p2[i]) - p2[i] for i in range(len(p2))]
elif len(p1) < len(p2):
    delta_center_arr = [find_nearest(p1, p2[i]) - p2[i] for i in range(len(p1))]
else: # If no
    delta_center_arr = [p2[i] - p2[i] for i in range(len(p1))]

print(f"Mean delta is {np.mean(delta_center_arr)}")
print(f"Standart error is: {np.std(delta_center_arr)}")
plt.plot(p2, delta_center_arr, label="Center shifts")
plt.xlabel("Wavelenght, Ang")
plt.ylabel("Nearby minimum diffirence")
plt.legend()
plt.show()
