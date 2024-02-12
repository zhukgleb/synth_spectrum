import matplotlib.pyplot as plt
import scienceplots
from velocity import find_velocity
import numpy as np

from data import extract_data


plt.style.use('science')

def make_report(spectrum, template, autocorr, corr):
    fig = plt.figure(figsize=(8, 8))
    v = 233.60411153029224
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(spectrum[:, 0], spectrum[:, 1], color='black', alpha=0.7)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.plot(template[:, 0], template[:, 1], color='black', alpha=0.7)
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax3.plot(corr[:, 0], corr[:, 1], color="black")
    ax3.plot([v, v], [-1, 1], "--", color='r')

    # Lims
    ax2.set_xlim((5750, 6000))
    ax3.set_xlim((v-1000*10*30,v + 1000*10*30))
    ax3.set_ylim((0, max(corr[:, 1])))
    # Titile
    ax1.set_title('Spectrum', loc='center')
    ax2.set_title('Template', loc='center')
    ax3.set_title('Correlation', loc='center')
    plt.show()




if __name__ == "__main__":
    spectrum = np.genfromtxt("results/spectrum.txt")
    template = np.genfromtxt("results/template.txt")
    autocorr = np.genfromtxt("results/autocorrelation.txt")
    corr = np.genfromtxt("results/correlation.txt")
    make_report(spectrum, template, autocorr, corr)
