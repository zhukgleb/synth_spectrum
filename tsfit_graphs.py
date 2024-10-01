import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def teff_graph(path2result: str):
    data = np.genfromtxt(path2result)
    data = data[(data[:, -1] == 0) & (data[:, -2] == 0)]  # Remove all warning and error lines
    teff = data[:, 1]
    teff_err = data[:, 2]
    wave_center = data[:, 3]
    ew = data[:, -3]
    rv = data[:, 4]
    with plt.style.context('science'):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.errorbar(teff, ew, xerr=teff_err, fmt="o",
                     color="black", alpha=0.8, elinewidth=1,
                       capsize=0)
        ax.set_xlabel(r"$T_{eff}$, K")
        ax.set_ylabel("EW")
        plt.show()
    print(data)




if __name__ == "__main__":
    t_path = "data/chem/02229_teff.dat"
    teff_graph(t_path)