import numpy as np
from tsfit_utils import get_model_data, clean_pd

localtsfit = True

def extract_linemask(path2linemask:str) -> np.ndarray:
    return np.genfromtxt(path2linemask)


# 
def clean_after_run(path2reuslts: str, path2linemask: str):
    linemask_data = extract_linemask(path2linemask)
    result_data = get_model_data(path2reuslts)
    centerline_pd = result_data["wave_start"].values
    cenerline_linemask = linemask_data[:, 0]

    


if __name__ == "__main__":
    from config_loader import tsfit_output
    path2output = "2025-02-22-16-06-06_0.8738030062131275_LTE_Fe_1D"
    clean_after_run(tsfit_output + path2output, "/Users/beta/synth_spectrum/linemask_files/Fe/fe1_gleb2.txt")
