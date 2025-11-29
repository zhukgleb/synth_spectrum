import numpy as np
import pandas as pd


def gauss():
    pass


def fit_line():
    pass


def load_linelist(element_name: str, wave_start: int = 4800, wave_end: int = 7800):
    filename = "atomic_lines.tsv"
    df = pd.read_csv(filename, sep="\t")
    condition = (
        (df["element"] == element_name)
        & (df["wave_A"] > wave_start)
        & (df["wave_A"] < wave_end)
    )
    df = df.loc[condition]
    print(df)


def main():
    spectra = np.genfromtxt("0_normalized.spec", comments="#")
    print(spectra)


if __name__ == "__main__":
    load_linelist("Fe 1")
