import numpy as np
import matplotlib.pyplot as plt


path2data = "data/Na5889.syn"

def extract_data(path=path2data):
    data = np.genfromtxt(path2data)
    ang = data[:, 0]
    red_flux = data[:, 1]
    cont = data[:, 3]
    ang.astype(float)
    return ang, red_flux, cont

if __name__ == "__main__":
    x ,y, _ = extract_data()
    plt.plot(x,y)
    plt.show()
