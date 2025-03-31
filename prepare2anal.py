from RV.bf import (
    load_spectra,
    interpolate_template,
    calculate_bf_profile,
    plot_bf_profile,
)
import numpy as np
import matplotlib.pyplot as plt
import PyAstronomy.pyasl as pyasl


def velocity_tilt(template, observed):
    c = 299792.458  # km/s
    interp_template = interpolate_template(observed, template)
    bf_profile = calculate_bf_profile(observed, interp_template, v_grid, c)
    # plot_bf_profile(v_grid, bf_profile)
    max_bf_index = np.where(bf_profile == max(bf_profile))
    velocity = v_grid[max_bf_index][0]
    print(f"velocity is {velocity}")
    _, observed[:, 0] = pyasl.dopplerShift(
        observed[:, 0], observed[:, 1], velocity, edgeHandling="firstlast"
    )
    return observed, velocity


template_path = "/Users/beta/synth_spectrum/data/iraszsynth.txt"
observed_path = "/Users/beta/synth_spectrum/data/iras05113_unshifted.txt"
wavelength_min, wavelength_max = 4700, 7777
v_grid = np.linspace(-60, 60, 600)
segment_width = 8
template, observed = load_spectra(
    template_path, observed_path, wavelength_min, wavelength_max
)
_, observed[:, 0] = pyasl.dopplerShift(
    observed[:, 0], observed[:, 1], 15, edgeHandling="firstlast"
)

velocity = 100

while abs(velocity) > 0.01:
    prev_iter = velocity
    observed, velocity = velocity_tilt(template, observed)
    if velocity == prev_iter:
        break

np.savetxt(observed_path + "_rest", observed)
