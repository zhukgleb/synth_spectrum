from velocity import find_velocity
import numpy as np

tsfit_input_path = "/home/lambda/code/synth_spectrum/iras2020.txt"
tsfit_synth = "/home/lambda/postagb.spec"

spectrum = np.genfromtxt(tsfit_input_path)
template = np.genfromtxt(tsfit_synth, skip_header=1)

# interval = [5200, 5968]
interval = [5200, 5300]

cv, zp, ze, sigma = find_velocity(
    [spectrum[:, 0], spectrum[:, 1]],
    [template[:, 0], template[:, 1]],
    interval,
    mult=100,
)
