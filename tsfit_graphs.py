import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from tsfit_utils import get_model_data
from typing import List, Union
from tsfit_utils import clean_pd
import scienceplots
from scipy.stats import norm
from scipy.optimize import curve_fit


def weighted_kde(x, weights, x_grid, bandwidth=0.1):
    kde_values = np.zeros_like(x_grid)
    for xi, wi in zip(x, weights):
        kde_values += wi * norm.pdf(x_grid, loc=xi, scale=bandwidth)
    return kde_values / kde_values.sum()


def teff_graph(path2result: str):
    data = np.genfromtxt(path2result)
    data = data[
        (data[:, -1] == 0) & (data[:, -2] == 0)
    ]  # Remove all warning and error lines
    teff = data[:, 1]
    teff_err = data[:, 2]
    ew = data[:, -3]
    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(4, 4))
        ax.errorbar(
            teff,
            ew,
            xerr=teff_err,
            fmt="o",
            color="black",
            alpha=0.8,
            elinewidth=1,
            capsize=0,
        )
        ax.set_xlabel(r"$T_{eff}$, K")
        ax.set_ylabel("EW")
        plt.show()
    print(data)


def plot_scatter_df_results(
    df_results: pd.DataFrame,
    x_axis_column: str,
    y_axis_column: str,
    xlim=None,
    ylim=None,
    color="black",
    invert_x_axis=False,
    invert_y_axis=False,
    **pltargs,
):
    if color in df_results.columns.values:
        pltargs["c"] = df_results[color]
        pltargs["cmap"] = "viridis"
        pltargs["vmin"] = df_results[color].min()
        pltargs["vmax"] = df_results[color].max()
        plot_colorbar = True
    else:
        pltargs["color"] = color
        plot_colorbar = False
    plt.scatter(df_results[x_axis_column], df_results[y_axis_column], **pltargs)
    plt.xlabel(x_axis_column)
    plt.ylabel(y_axis_column)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if invert_x_axis:
        plt.gca().invert_xaxis()
    if invert_y_axis:
        plt.gca().invert_yaxis()
    if plot_colorbar:
        # colorbar with label
        plt.colorbar(label=color)
    plt.show()
    plt.close()


def plot_metall(data: pd.DataFrame, ratio: str = "Fe_H"):
    metallicity = data[ratio].to_numpy(float)
    error = data["chi_squared"].to_numpy(float)

    xy_point_density = np.vstack([metallicity, error])
    z_point_density = gaussian_kde(xy_point_density)(xy_point_density)
    idx_sort = z_point_density.argsort()
    x_plot, y_plot, z_plot = (
        metallicity[idx_sort],
        error[idx_sort],
        z_point_density[idx_sort],
    )

    weights = 1 / error
    weighted_avg = np.sum(metallicity * weights) / np.sum(weights)

    median = np.median(metallicity)

    lower_bound = np.percentile(metallicity, 2.5)
    upper_bound = np.percentile(metallicity, 97.5)

    print(f"Взвешенное среднее: {weighted_avg:.3f}")
    print(f"Медиана: {median:.3f}")
    print(f"95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")

    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"Metallicity IRAS Z02229+6208")
        ax.set_ylabel(r"$\chi^{2}$")
        ax.set_xlabel(r"Metallicity, [Fe/H]")
        ax.set_ylim((0, 10))
        density = ax.scatter(x_plot, y_plot, c=z_plot)
        plt.colorbar(density)
        plt.show()


def plot_ion_balance(data: pd.DataFrame):
    metallicity = data["Fe_H"].to_numpy(float)
    ew = data["ew"].to_numpy(float)
    ew = ew
    lamb = data["wave_center"].to_numpy(float)
    rel_ew = ew / lamb

    # Regression
    x = rel_ew.reshape((-1, 1))
    y = metallicity
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_
    slope = slope[0]
    intercept = model.intercept_
    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("slope:", model.coef_)
    xfit = np.linspace(0, 100, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"Ionization balance of IRAS Z02229+6208")
        ax.set_ylabel(r"Metallicity, [Fe/H]")
        ax.set_xlabel(r"$EW / \lambda$")
        ax.set_ylim((-0.5, 0))
        # ax.set_xscale("log")
        text = f"Slope: {slope:.4f}\nIntercept: {intercept:.2f}\nR squared: {r_sq:.2f}"
        plt.annotate(
            text,
            xy=(0.95, 0.95),  # upper left corner
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
            ),
        )
        plt.scatter(
            rel_ew, metallicity, color="black", alpha=0.5, label="derived metallicity"
        )
        plt.plot(xfit, yfit, label="linear regression")
        plt.legend()
        plt.show()


def plot_metallVS(data_1: pd.DataFrame, data_2: pd.DataFrame, ratio: str = "Fe_H"):
    metallicity_1 = data_1[ratio].to_numpy(float)
    error_1 = data_1["chi_squared"].to_numpy(float)

    xy_point_density_1 = np.vstack([metallicity_1, error_1])
    z_point_density_1 = gaussian_kde(xy_point_density_1)(xy_point_density_1)
    idx_sort_1 = z_point_density_1.argsort()
    x_plot_1, y_plot_1, z_plot_1 = (
        metallicity_1[idx_sort_1],
        error_1[idx_sort_1],
        z_point_density_1[idx_sort_1],
    )

    metallicity_2 = data_2[ratio].to_numpy(float)
    print(np.std(metallicity_2))
    error_2 = data_2["chi_squared"].to_numpy(float)

    xy_point_density_2 = np.vstack([metallicity_2, error_2])
    z_point_density_2 = gaussian_kde(xy_point_density_2)(xy_point_density_2)

    idx_sort_2 = z_point_density_2.argsort()
    x_plot_2, y_plot_2, z_plot_2 = (
        metallicity_2[idx_sort_2],
        error_2[idx_sort_2],
        z_point_density_2[idx_sort_2],
    )

    with plt.style.context("science"):
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 9))
        # ax[0].set_title(r"Metallicity IRAS Z02229+6208")
        ax[0].set_title(r"Metallicity IRAS 07430+1115, T=6000, log~g=1")

        ax[0].set_ylabel(r"$\chi^{2}$")
        ax[0].set_xlabel(r"Metallicity, [Fe/H]")
        ax[0].set_ylim((0, 100))
        ax[1].set_title(r"Metallicity IRAS 07430+1115, T=4900, log~g=0.5")
        ax[1].set_ylabel(r"$\chi^{2}$")
        ax[1].set_xlabel(r"Metallicity, [Fe/H]")
        ax[1].set_ylim((0, 100))
        density_1 = ax[0].scatter(x_plot_1, y_plot_1, c=z_plot_1)
        density_2 = ax[1].scatter(x_plot_2, y_plot_2, c=z_plot_2)

        plt.colorbar(density_2)
        plt.colorbar(density_1)
        # plt.savefig("ReddyVSZhuck.pdf", dpi=600)
        plt.show()


def hist_estimation(df, bins):
    metall = "Fe_H"
    # b = df.iloc[:, 1:].values
    counts, bins = np.histogram(pd.to_numeric(df[metall]), 30)
    print(counts)
    sigma = (max(bins) ** 0.5) / (
        (bins[-1] - bins[-2]) * len(pd.to_numeric(df[metall]))
    )
    print(sigma)
    plt.stairs(counts, bins)
    # plt.hist(df[metall], bins, histtype="bar", alpha=0.5)
    # plt.xlabel(metall)
    # plt.ylabel("Count")
    plt.show()


def plot_metall_error(data: pd.DataFrame):
    metallicity = data["Fe_H"].to_numpy(float)
    error = data["chi_squared"].to_numpy(float)
    kde = gaussian_kde(metallicity, bw_method="scott")  # bw_method можно настроить
    x_grid = np.linspace(metallicity.min() - 0.05, metallicity.max() + 0.05, 1000)
    pdf = kde(x_grid)  # Плотность вероятности

    mode = x_grid[np.argmax(pdf)]

    cdf = np.cumsum(pdf) / np.sum(pdf)  # Нормированная кумулятивная функция
    lower_bound = x_grid[np.searchsorted(cdf, 0.025)]
    upper_bound = x_grid[np.searchsorted(cdf, 0.975)]

    # Визуализация
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, pdf, label="KDE (оценка плотности)", color="blue")
    plt.axvline(mode, color="red", linestyle="--", label=f"Мода: {mode:.3f}")
    plt.axvline(
        lower_bound, color="green", linestyle="--", label=f"2.5%: {lower_bound:.3f}"
    )
    plt.axvline(
        upper_bound, color="green", linestyle="--", label=f"97.5%: {upper_bound:.3f}"
    )
    plt.title("Ядровая оценка плотности (KDE)")
    plt.xlabel("Металличность")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    print(f"Мода металличности: {mode:.3f}")
    print(f"95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")
    plt.show()


def plot_metall_KDE(data: pd.DataFrame, ratio: str = "Fe_H", bandwidth=0.05):
    metallicity = data[ratio].to_numpy(float)
    print(f"Mean: {np.mean(metallicity)}")
    chi_squared = data["chi_squared"].to_numpy(float)
    weights = 1 / chi_squared
    weights /= weights.sum()

    x_grid = np.linspace(metallicity.min() - 0.05, metallicity.max() + 0.05, 1000)
    pdf = weighted_kde(metallicity, weights, x_grid, bandwidth=bandwidth)

    mode = x_grid[np.argmax(pdf)]

    cdf = np.cumsum(pdf) / np.sum(pdf)
    lower_bound = x_grid[np.searchsorted(cdf, 0.025)]
    upper_bound = x_grid[np.searchsorted(cdf, 0.975)]
    with plt.style.context("ggplot"):
        plt.figure(figsize=(8, 6))
        plt.plot(x_grid, pdf, label="KDE", color="blue")
        plt.axvline(mode, color="red", linestyle="--", label=f"Mode: {mode:.3f}")
        plt.axvline(
            lower_bound, color="green", linestyle="--", label=f"2.5%: {lower_bound:.3f}"
        )
        plt.axvline(
            upper_bound,
            color="green",
            linestyle="--",
            label=f"97.5%: {upper_bound:.3f}",
        )
        plt.title("KDE with weight")
        plt.xlabel("Metallicity")
        plt.ylabel("PDF")
        plt.legend()
        plt.show()

        print(f"Мода металличности: {mode:.3f}")
        print(f"95% доверительный интервал: [{lower_bound:.3f}, {upper_bound:.3f}]")
        # 68 % interval
        lower_bound_68 = x_grid[np.searchsorted(cdf, 0.16)]  # 16-й процентиль
        upper_bound_68 = x_grid[np.searchsorted(cdf, 0.84)]  # 84-й процентиль

        error = (upper_bound_68 - lower_bound_68) / 2
        print(f"Средняя ошибка металличности (1σ): {error:.3f}")


def median_analysis(pd_data: pd.DataFrame):
    column_name = "Teff"
    if np.argwhere(pd_data.columns.values == column_name) != -1:
        column_data = pd_data[column_name].values
        column_data = [float(column_data[x]) for x in range(len(column_data))]
        column_data_median = np.median(column_data)
        with plt.style.context("science"):
            plt.title(f"{column_name} variation")
            plt.scatter([x for x in range(len(column_data))], column_data)
            plt.show()

        print(f"Median vmic: {column_data_median}")
        bootstrapped_medians = [
            np.median(
                np.random.choice(column_data, size=len(column_data), replace=True)
            )
            for _ in range(10**6)
        ]
        median_variance = np.var(bootstrapped_medians)
        print(f"Дисперсия медианы: {median_variance}")


def gaussian(Z, A, mu, sigma):
    return A * np.exp(-((Z - mu) ** 2) / (2 * sigma**2))


def gauss_metall(pd_data: pd.DataFrame, ratio: str = "Fe_H"):
    feh_values = pd_data[ratio].to_numpy(float)
    chi2_values = pd_data["chi_squared"].to_numpy(float)
    sorted_idx = np.argsort(feh_values)
    feh_values = feh_values[sorted_idx]
    chi2_values = chi2_values[sorted_idx]
    # Найдём минимум Chi^2
    min_index = np.argmin(chi2_values)
    feh_best = feh_values[min_index]
    chi2_min = chi2_values[min_index]

    # Оценим погрешность по критерию Delta Chi^2 = 1
    threshold = chi2_min + 1

    # Определим границы неопределенности
    feh_lower = np.max(
        feh_values[feh_values < feh_best][
            chi2_values[feh_values < feh_best] <= threshold
        ],
        initial=feh_best,
    )
    feh_upper = np.min(
        feh_values[feh_values > feh_best][
            chi2_values[feh_values > feh_best] <= threshold
        ],
        initial=feh_best,
    )

    # Вывод результатов
    print(f"Оптимальное значение [Fe/H]: {feh_best:.3f}")
    print(
        f"Неопределенность: -{feh_best - feh_lower:.3f} / +{feh_upper - feh_best:.3f}"
    )

    # Визуализация
    plt.plot(feh_values, chi2_values, "o-", label="Chi^2")
    plt.axhline(threshold, linestyle="--", color="gray", label="$\chi^2_{min} + 1$")
    plt.axvline(
        feh_best, linestyle="--", color="red", label=f"Best [Fe/H] = {feh_best:.3f}"
    )
    plt.fill_betweenx(
        [min(chi2_values), max(chi2_values)],
        feh_lower,
        feh_upper,
        color="gray",
        alpha=0.3,
    )
    plt.xlabel("[Fe/H]")
    plt.ylabel("$\chi^2$")
    plt.legend()
    plt.show()


def velocity_dispersion(pd_data: pd.DataFrame, ratio: str = "Fe_H"):
    velocity = pd_data["Doppler_Shift_add_to_RV"].values
    velocity = [float(velocity[x]) for x in range(len(velocity))]
    metallicty = pd_data[ratio].values
    metallicty = [float(metallicty[x]) for x in range(len(metallicty))]
    chi = pd_data["chi_squared"].to_numpy(float)

    with plt.style.context("science"):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        ax.set_title(r"Velocity dispersion of FeI on IRAS 07430+1115")
        ax.set_ylabel(r"$\chi^{2}$")
        ax.set_xlabel(r"Metallicity, [Fe/H]")
        ax.set_ylim((0, 20))

        density = ax.scatter(metallicty, chi, c=velocity)
        plt.colorbar(density, label="Velocity dispersion")
        # plt.savefig("ReddyVSZhuck.pdf", dpi=600)
        plt.show()

    pass


def wave2velocity(wavelength: float, rest_wavelength: float) -> float:
    z = (wavelength - rest_wavelength) / wavelength
    c = 2.998e8  # m/s
    return float(z * c)


def line_combiner(spectrum: np.ndarray, linelist: np.ndarray):
    lines_cut = []
    for line in linelist:
        line_data = np.where((spectrum[:, 0] >= line[1]) & (spectrum[:, 0] <= line[2]))
        lines_cut.append(spectrum[line_data])

    velocity_cut = []
    number_of_line = 0
    for line in lines_cut:
        line_velocity = []
        for element in line:
            line_velocity.append(wave2velocity(element[0], linelist[number_of_line][0]))
        number_of_line += 1
        print(number_of_line)
        velocity_cut.append(line_velocity)

    for i in range(len(velocity_cut)):
        plt.plot(velocity_cut[i], lines_cut[i][:, 1])
        # plt.plot(np.arange(0, len(lines_cut[i][:, 1])), lines_cut[i][:, 1])

    plt.show()


if __name__ == "__main__":
    from tsfit_utils import get_model_data
    from config_loader import tsfit_output

    #    out_1 = "2025-02-11-17-09-42_0.04291624334452615_LTE_Fe_1D"
    #    out_2 = "2025-02-10-11-48-42_0.616663993169006_LTE_Fe_1D"
    #
    out_1 = "2025-02-22-16-06-06_0.8738030062131275_LTE_Fe_1D"
    out_2 = "2025-02-22-14-19-48_0.23837068316608268_LTE_Fe_1D"

    pd_data_1 = get_model_data(tsfit_output + out_1)
    pd_data_2 = get_model_data(tsfit_output + out_2)
    # line_combiner(spectrum, c_linemask)

    # pd_data_1 = clean_pd(pd_data_1, True, True)
    # pd_data_2 = clean_pd(pd_data_2, True, True)

    r = "Fe_H"
    # velocity_dispersion(pd_data_2, r)
    # plot_metallVS(pd_data_1, pd_data_2, r)
    plot_metall_KDE(pd_data_1, r)
    plot_metall_KDE(pd_data_2, r)

    # median_analysis(pd_data_2)
    # plot_ion_balance(pd_data_2)
    # hist_estimation(pd_data_2, 30)
