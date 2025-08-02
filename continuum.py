import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    spectrum = np.genfromtxt("data/0.spec")

    wave = spectrum[:, 0]
    norm_flux = spectrum[:, 1]
    unnorm_flux = spectrum[:, 2]

    plt.plot(wave, norm_flux)
    plt.plot(wave, unnorm_flux)
    plt.show()
