import numpy as np
from pandas.core.frame import fmt
from tsfit_utils import get_model_data, clean_pd

localtsfit = True

def extract_linemask(path2linemask:str) -> np.ndarray:
    return np.genfromtxt(path2linemask)

# def clean_after_run(path2reuslts: str, path2linemask: str):
#     linemask_data = extract_linemask(path2linemask)
#     result_data = get_model_data(path2reuslts)
#     centerline_pd = result_data["wave_center"].astype(float)
    
#     cenerline_linemask = linemask_data[:, 0]
#     idx = np.where(cenerline_linemask == centerline_pd)
#     print(cenerline_linemask, centerline_pd)
#     print(idx)
    

def clean_after_run(path2reuslts: str, path2linemask: str, truncate_warnings: bool = True):
    linemask_data = extract_linemask(path2linemask)
    result_data = get_model_data(path2reuslts)
    print(f"Input data contain a {len(result_data)} entries")
    result_data = clean_pd(result_data, truncate_warnings, True)
    print(f"After cleaning a {len(result_data)} entries")

    centerline_pd = result_data["wave_center"].astype(float)
    startline_pd = result_data["wave_start"].astype(float)
    endline_pd = result_data["wave_end"].astype(float)

    clean_linelist = np.column_stack((centerline_pd, startline_pd, endline_pd))
    """
    Я не совсем понимаю почему, но чистый лайнлист все равно продуцировал ошибки. Пока не знаю, с чем это
    свзанно, так что оставлю здесь этот комментарий, потому что явно есть над чем подумать..
    """
    np.savetxt(path2linemask + "_clean", clean_linelist, fmt="%s")
    

def extract_element(path2vald: str, element_name: str) -> np.ndarray:
    data = np.genfromtxt(path2vald, delimiter=",", dtype=str, skip_header=3, invalid_raise=False)
    idx = np.where(data[:, 0] == element_name)
    return data[:, 1]


if __name__ == "__main__":
    from config_loader import tsfit_output
    # path2output = "2025-02-22-16-06-06_0.8738030062131275_LTE_Fe_1D"
    # clean_after_run(tsfit_output + path2output, "/Users/beta/synth_spectrum/linemask_files/Fe/fe1_gleb2.txt")
    element_data = extract_element("ZhuklevichGleb.006110", "N 1")
    print(element_data)