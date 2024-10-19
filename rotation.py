import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
        print("Фитирование не удалось.")
        return 0, 0
    except ValueError:
        print("Value error")
        return 0, 0
    
    # Извлечение параметра sigma для вычисления FWHM
    _, _, sigma = popt
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma  # Формула FWHM для гауссианы
    
    return sigma, perr[2]




if __name__ == "__main__":
    path2spectrum = "iras_2020.txt"
    path2linemask = "linemask_fe.txt"

    # 1th column is wavelenght, 2th -- relative intens
    spectrum_data = np.genfromtxt(path2spectrum)
    # 1th column left wing of line, 2th -- center and 3th are right wing
    linemask = np.genfromtxt(path2linemask)
    sigma, sigma_err = [], []
    x_list = []
    for i in range(len(linemask)):
        s, s_err = calculate_fwhm_absorption_fit(spectrum_data, linemask[i])
        sigma.append(s)
        sigma_err.append(s_err)
        x_list.append(i)


    plt.errorbar(x_list, sigma, yerr=sigma_err, fmt="o")
    plt.show()
    # print(fwhm)

