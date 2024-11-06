import numpy as np
import pandas as pd
import os


def get_model_data(path2model: str) -> pd.DataFrame:
    path2output = path2model + "/output"  # rewrite for NOT linux only
    with open(os.path.join(path2output), "r") as output_file:
        output_file_lines = output_file.readlines()

    output_file_header = output_file_lines[0].strip().split("\t")
    output_file_header[0] = output_file_header[0].replace("#", "")
    output_file_data_lines = [line.strip().split() for line in output_file_lines[1:]]
    output_file_df = pd.DataFrame(output_file_data_lines, columns=output_file_header)

    return output_file_df


# delete all bad lines after fitting (old function)
def clean_linemask(
    path2linemask: str, path2output: str, delete_warnings=False, delete_errors=True
):
    output_file_df = get_model_data(path2output)
    # df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]
    output_file_df["chi_squared"] = pd.to_numeric(output_file_df["chi_squared"])
    clear_df = output_file_df.loc[
        (output_file_df["flag_error"] == "00000000")
        & (output_file_df["chi_squared"] < 1)
        & (output_file_df["flag_warning"] == "00000000")
    ]
    print(clear_df)
    linemask = clear_df[["wave_start", "wave_center", "wave_end"]]
    np.savetxt(path2linemask + ".clear", linemask.values, fmt="%s")
    print(linemask)


def clean_pd(
    pd_data: pd.DataFrame, remove_warnings=False, remove_errors=True
) -> pd.DataFrame:
    pd_data["chi_squared"] = pd.to_numeric(pd_data["chi_squared"])
    output_data = pd.DataFrame()
    if remove_errors:
        output_data = pd_data.loc[
            (pd_data["flag_error"] == "00000000") & (pd_data["chi_squared"] < 10)
        ]
    if remove_warnings:
        output_data = pd_data.loc[
            (pd_data["flag_warning"] == "00000000") & (pd_data["chi_squared"] < 10)
        ]

    return output_data


def get_spectra(path2output) -> np.ndarray:
    # First things First
    # We have a two types spectra in folder -- one is real, observed.
    # and second (actualy, there is two of them) synthetic
    # synthetic have a two versions -- convoled with some R
    # and raw, without any scrab. with flux (!)

    files_in_output = os.listdir(path2output)
    for file in files_in_output:
        if file.find("convolved") != -1:
            convolved_spectra_name = file
        if file.find("convolved") == -1 & file.find("result_spectrum"):
            synth_spectra_name = file

    print(synth_spectra_name, convolved_spectra_name)
    output = np.ndarray([1, 2, 3])
    return output


if __name__ == "__main__":
    data_path = "/home/gamma/TSFitPy/output_files/5100_05_VGFe12/"
    linemask_path = (
        "/home/gamma/TSFitPy/input_files/linemask_files/Fe/fe12-lmask_VG_clear.txt"
    )
    # clean_linemask(linemask_path, data_path)
    get_spectra(data_path)
