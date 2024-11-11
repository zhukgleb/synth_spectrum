from velocity import find_velocity
import numpy as np
import matplotlib.pyplot as plt

tsfit_input_path = "/home/lambda/TSFitPy/input_files/observed_spectra/iras2020.txt"
tsfit_synth = "/home/lambda/postagb.spec"

spectrum = np.genfromtxt(tsfit_input_path)
template = np.genfromtxt(tsfit_synth, skip_header=1)

# plt.plot(spectrum[:, 0], spectrum[:, 1])
# plt.plot(template[:, 0], template[:, 1])
# plt.show()

interval = [5200, 5968]

cv, zp, ze, sigma = find_velocity(
    [spectrum[:, 0], spectrum[:, 1]], template, interval, mult=10
)
