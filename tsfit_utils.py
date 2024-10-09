import numpy as np
import pandas as pd
import os

# delete all bad lines after fitting
def clean_linemask(path2linemask: str, path2output: str, 
                   delete_warnings=False, delete_errors=True):
    
    with open(os.path.join(path2output), 'r') as output_file:
        output_file_lines = output_file.readlines()

    output_file_header = output_file_lines[0].strip().split('\t')
    output_file_header[0] = output_file_header[0].replace("#", "")
    output_file_data_lines = [line.strip().split() for line in output_file_lines[1:]]
    output_file_df = pd.DataFrame(output_file_data_lines, columns=output_file_header)
    # df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]
    output_file_df['chi_squared'] = pd.to_numeric(output_file_df['chi_squared'])
    clear_df = output_file_df.loc[(output_file_df['flag_error'] == "00000000") & (output_file_df['chi_squared'] < 1)]
    print(clear_df)
    linemask = clear_df[["wave_start", "wave_center", "wave_end"]]
    np.savetxt(path2linemask + ".clear", linemask.values, fmt="%s")
    print(linemask)



if __name__ == "__main__":
    data_path = "/home/alpha/TSFitPy/output_files/Oct-08-2024-00-24-16_0.8690955000098243_LTE_Fe_1D/output"
    linemask_path = "/home/alpha/TSFitPy/input_files/linemask_files/Fe/fe1-lmask.txt"
    clean_linemask(linemask_path, data_path)