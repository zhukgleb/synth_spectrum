import matplotlib.pyplot as plt
from velocity import find_velocity
import numpy as np

from data import extract_data



def make_report(spectrum, template, autocorr, corr):
    plt.figure(figsize=(8,8))
    pass



if __name__ == "__main__":
    spectrum = np.genfromtxt("results/spectrum.txt")
    template = np.genfromtxt("results/template.txt")
    autocorr = np.genfromtxt("results/autocorrelation.txt")
    corr = np.genfromtxt("results/correlation.txt")
    make_report(spectrum, template, autocorr, corr)
