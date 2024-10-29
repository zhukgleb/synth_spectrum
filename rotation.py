import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots


def inverted_gaussian(x, amp, mu, sigma):
    return 1 - amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def calculate_abs_params(spectrum: np.ndarray, linemask: list) -> list:
    wavelength = spectrum[:, 0]
    intens = spectrum[:, 1]
    left_wing, line_center, right_wing = linemask
    # need to cut spectrum out of line
    # 4840.8
    plus_rv = 0.0
    crop_mask = np.where(
        (wavelength >= left_wing + plus_rv) & (wavelength <= right_wing + plus_rv)
    )
    wavelength = wavelength[crop_mask]
    intens = intens[crop_mask]

    # make a synth curve

    # x = np.arange(min(wavelength), max(wavelength), 0.001)
    # y = inverted_gaussian(x, 0.5, 4840.8, 0.02)
    # y += np.random.normal(loc=0.0, scale=1/100, size=len(x))
    # plt.plot(x, y)
    # popt, pcov = curve_fit(inverted_gaussian, x, y, p0=[np.abs(min(y)), line_center, (right_wing - left_wing) / 2.0])
    # plt.plot(x, inverted_gaussian(x, popt[0], popt[1], popt[2]))
    # plt.show()
    try:
        initial_guess = [
            np.abs(min(wavelength)),
            line_center,
            (right_wing - left_wing) / 2.0,
        ]
    except ValueError:
        initial_guess = [0, 0, 0]
    try:
        popt, pcov = curve_fit(
            inverted_gaussian, wavelength, intens, p0=initial_guess, maxfev=50000
        )
        perr = np.sqrt(np.diag(pcov))
        print(f"{line_center} ok")
    except RuntimeError:
        print(f"{line_center} not ok")
        return [-1, -1, -1], [-2, -2, -2]
    except ValueError:
        print(f"{line_center} not ok (value)")
        return [-2, -2, -2], [-2, -2, -2]

    # fit_x = np.arange(min(wavelength), max(wavelength), 0.001)
    # fit_y = inverted_gaussian(fit_x, popt[0], popt[1], popt[2])
    # plt.plot(wavelength, intens)
    # plt.plot(fit_x, fit_y)
    # plt.show()

    return popt, perr


if __name__ == "__main__":
    path2spectrum = "iras_2020.txt"
    path2linemask = "linemask_fe.txt"

    # 1th column is wavelength, 2th -- relative intens
    spectrum_data = np.genfromtxt(path2spectrum)
    # 1th column left wing of line, 2th -- center and 3th are right wing
    linemask = np.genfromtxt(path2linemask)
    params_arr = []
    for i in range(len(linemask)):
        params_arr.append(calculate_abs_params(spectrum_data, linemask[i]))

    popt, perr = list(zip(*params_arr))
    depth, rv, sigma = list(zip(*popt))
    ddepth, drv, dsigma = list(zip(*perr))

    depth = np.array(depth)
    rv = np.array(rv)
    sigma = np.array(sigma)
    ddepth = np.array(ddepth)
    drv = np.array(drv)
    dsigma = np.array(dsigma)

    good_indexes = np.where(
        (sigma * 2.335 > 0)
        & (sigma * 2.335 < 5)
        & (dsigma * 2.335 < 1)
        & (dsigma * 2.335 > 0) * (depth < 0.9)
    )
    depth = depth[good_indexes]
    rv = rv[good_indexes]
    sigma = sigma[good_indexes]
    ddepth = ddepth[good_indexes]
    drv = drv[good_indexes]
    dsigma = dsigma[good_indexes]
    save = True
    if save:
        sigma = sigma * 2.335
        dsigma = dsigma * 2.235
        np.savetxt("fwhm_data.txt", np.column_stack((rv, sigma, depth, dsigma)))

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"FWHM versus wavelength, IRAS Z02229+6208")
        ax.set_ylabel("FWHM")
        ax.set_xlabel(r"Wavelength, \AA")
        ax.errorbar(
            rv,
            sigma * 2.335,
            yerr=dsigma * 2.335,
            fmt="none",
            ecolor="black",
            alpha=0.8,
            mew=4,
        )
        sc = ax.scatter(rv, sigma * 2.335, c=depth, cmap="plasma")
        plt.colorbar(sc, label="Depth")
        if save:
            plt.savefig("rotation.pdf", dpi=150)
        else:
            plt.show()
