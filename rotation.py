import numpy as np

if __name__ == "__main__":
    path2spectrum = "iras2020.txt"
    path2linemask = "linemask_fe.txt"

    # 1th column is wavelenght, 2th -- relative intens
    spectrum_data = np.genfromtxt(path2spectrum)
    # 1th column left wing of line, 2th -- center and 3th are right wing
    linemask = np.genfromtxt(path2linemask)