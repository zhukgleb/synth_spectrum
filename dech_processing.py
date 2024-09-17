"""
A small module of various functions for working with dech20/95 output files
"""

from numpy import genfromtxt, concatenate, column_stack, savetxt, ndarray
from astropy.io import fits

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
    print(hdu_list.info())
    image_data = [hdu_list[x].data for x in range(len(hdu_list))]
    return image_data[0]


if __name__ == "__main__":
    # data = tab_spectra("dech30.tab")
    # print(data)
    # data = dech_fits_loader("data/fits/e718009s.fits")