import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots

# Определение функции инвертированной гауссианы для абсорбции
def inverted_gaussian(x, amp, mu, sigma):
    return -amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Функция для вычисления FWHM через фитирование инвертированной гауссианой
def calculate_fwhm_absorption_fit(spectrum, line_mask):
    # Извлечение данных
    wavelength = spectrum[:, 0]
    intensity = spectrum[:, 1]
    
    # Установка маски линии
    left_wing, line_center, right_wing = line_mask
    
    # Фильтрация данных внутри маски
    mask = (wavelength >= left_wing) & (wavelength <= right_wing)
    wavelength_line = wavelength[mask]
    intensity_line = intensity[mask]
    
    # Начальные значения для параметров [амплитуда, центр, стандартное отклонение]
    # Амплитуда инвертированной гауссианы положительная, так как функция умножается на -1
    try:
        initial_guess = [np.abs(np.min(intensity_line)), line_center, (right_wing - left_wing) / 2.0]
    except ValueError:
        initial_guess = [0, 0, 0]
    
    # Аппроксимация данных инвертированной гауссианой
    try:
        popt, pcov = curve_fit(inverted_gaussian, wavelength_line, intensity_line, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        # print("Фитирование не удалось.")
        return 0, 0, 0, 0
    except ValueError:
        # print("Value error")
        return 0, 0, 0, 0
    
    # Извлечение параметра sigma для вычисления FWHM
    depth, rv, sigma = popt
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma  # Формула FWHM для гауссианы
    
    return fwhm, depth, rv, 2 * np.sqrt(2 * np.log(2)) * perr[2]




if __name__ == "__main__":
    path2spectrum = "iras_2020.txt"
    path2linemask = "linemask_fe.txt"

    # 1th column is wavelenght, 2th -- relative intens
    spectrum_data = np.genfromtxt(path2spectrum)
    # 1th column left wing of line, 2th -- center and 3th are right wing
    linemask = np.genfromtxt(path2linemask)
    fwhm, depth, rv,  fwhm_err = [], [], [], []
    for i in range(len(linemask)):
        s, d, r,  s_err = calculate_fwhm_absorption_fit(spectrum_data, linemask[i])
        fwhm.append(s)
        depth.append(d)
        fwhm_err.append(s_err)
        rv.append(r)  # Not good name...
    
    fwhm = np.array(fwhm)
    fwhm_err = np.array(fwhm_err)
    depth = np.array(depth)
    rv = np.array(rv)

    good_indexes = np.where((fwhm > 0) & (fwhm < 5) & (fwhm_err < 1))
    fwhm = fwhm[good_indexes]
    fwhm_err = fwhm_err[good_indexes]
    depth = depth[good_indexes] * -1
    rv = rv[good_indexes]
    lm = linemask[good_indexes]

    # x_val = [x for x in range(len(fwhm))]
    lm = [lm[x][1] for x in range(len(fwhm))]
    with plt.style.context('science'):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"FWHM versus wavelength, IRAS Z02229+6208")
        ax.set_ylabel("FWHM")
        ax.set_xlabel(r"Wavelength, \AA")
        ax.errorbar(lm, fwhm, yerr=fwhm_err, fmt="none", ecolor="black", alpha=0.8, mew=4)
        sc = ax.scatter(lm, fwhm, c=depth, cmap="plasma")
        plt.colorbar(sc, label="Depth")
        plt.show()