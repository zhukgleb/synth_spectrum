import numpy as np


def load_linelist(element_name):
    pass

def main():
    spectra = np.genfromtxt("0_normalized.spec", comments='#')
    print(spectra)


if __name__ == "__main__":
    main()