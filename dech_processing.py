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

def fds_loader(path2data: str):
    orders_num = 40
    values_per_block = 2048
    block_size = 4
    total_values = orders_num * values_per_block

    # Чтение файла и извлечение данных
    with open(path2data, 'rb') as file:
        file_data = file.read()

    expected_size = total_values * block_size
    if len(file_data) < expected_size:
        raise ValueError("Размер файла меньше ожидаемого для данной структуры, поправьте кол-во порядков/формат записи")

    format_string = '<' + 'i' * total_values  # '<' для little-endian, 'i' для 32-битного целого числа
    unpacked_data = struct.unpack(format_string, file_data[:expected_size])

    orders = [unpacked_data[i:i + values_per_block] for i in range(0, total_values, values_per_block)]

    # print("Первый блок:", orders[0])
    print(f"start wavelenght: {orders[0][0]}")
    print(f"start wavelenght: {orders[-1][-1]}")

    return orders

if __name__ == "__main__":
    # data = tab_spectra("dech30.tab")
    # print(data)
    data = fds_loader("data/fits/s693011s.fds")
    