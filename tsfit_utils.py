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

    clear_df = output_file_df.loc[output_file_df['flag_error'] == "00000000"]
    linemask = clear_df[["wave_start", "wave_center", "wave_end"]]
    np.savetxt(path2linemask + ".clear", linemask.values, fmt="%s")
    print(linemask)



if __name__ == "__main__":
    data_path = "data/chem/output"
    linemask_path = "/home/alpha/TSFitPy/input_files/linemask_files/Fe/fe-lmask_VG.txt"
    clean_linemask(linemask_path, data_path)
