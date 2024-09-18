"""
A small module of various functions for working with dech20/95 output files
"""

from numpy import genfromtxt, concatenate, column_stack, savetxt, ndarray
from astropy.io import fits
import struct


"""
There in Dech is an option to choose to save all orders at once into one file,
but in this case all orders will not be saved in order,
but written down in a column, which creates a little confusion
"""
def tab_spectra(filepath: str, save=False) -> ndarray:
    data = genfromtxt(filepath)
    all_wave = []
    all_flux = []
    
    for i in range(0, len(data[0]), 2):
        all_wave.append(data[:, i])

    for i in range(1, len(data[0]), 2):
        all_flux.append(data[:, i])

    conc_wave = concatenate(all_wave)
    conc_flux = concatenate(all_flux)
    output =  column_stack((conc_wave, conc_flux))
    
    if save:
        savetxt("filepath" + "concat.txt", output)
    else:
        return output


"""
load a dech-specif. fits file
"""
def dech_fits_loader(path2data: str):
    hdu_list = fits.open(path2data)
    # print(hdu_list[0].header)
    image_data = [hdu_list[x].data for x in range(len(hdu_list))]
    return image_data[0]


""""
Сonverts from a hex file fds to a decimal one.
Of the useful and necessary -- this is the number of orders,
which affects the final extraction. It is also important
to control the binning, for example,
for 2 by 2 the block size will already be 1024,
and the number of orders will drop by half
"""
def fds_loader(path2data: str, orders_num=40, values_per_block=2048):
    block_size = 4
    total_values = orders_num * values_per_block

    with open(path2data, 'rb') as file:
        file_data = file.read()

    expected_size = total_values * block_size
    if len(file_data) < expected_size:
        raise ValueError("Expetred size is larger, than expected")

    format_string = '<' + 'i' * total_values 
    unpacked_data = struct.unpack(format_string, file_data[:expected_size])

    orders = [unpacked_data[i:i + values_per_block] for i in range(0, total_values, values_per_block)]
    for i in range(len(orders)):  # Yep, i don't know how this make in list generator correctrly....
        orders[i] = list(orders[i])  # Not good cast
        for j in range(len(orders[i])):
            orders[i][j] = orders[i][j] * 10**-4  # in angstroms

    print(f"start wavelenght: {orders[0][0]}")
    print(f"start wavelenght: {orders[-1][-1]}")

    return orders


def make_txt_from_spectra(path2spectra: str, path2fds: str):
    intens_vals = dech_fits_loader(path2spectra)
    ang_vals =  fds_loader(path2fds)

    return [intens_vals, ang_vals]

if __name__ == "__main__":
    # data = tab_spectra("dech30.tab")
    # print(data)
    # data = fds_loader("data/fits/s693011s.fds")
    data = make_txt_from_spectra("data/fits/e693006s.fits.200", "data/fits/s693011s.fds")
    print(data)