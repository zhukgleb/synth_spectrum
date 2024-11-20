import numpy as np
import matplotlib.pyplot as plt

template = np.loadtxt("/home/lambda/postagb.spec")
observed = np.loadtxt("/home/lambda/TSFitPy/input_files/observed_spectra/iras2020.txt")

interp_template = np.interp(observed[:, 0], template[:, 0], template[:, 1])

diff_map = observed[:, 1] - interp_template

plt.plot(observed[:, 0], diff_map)
plt.plot(observed[:, 0], observed[:, 1], color="crimson", alpha=0.5, label="observed")
plt.plot(template[:, 0], template[:, 1], color="black", alpha=0.5, label="synthetic")
plt.legend()
plt.show()
