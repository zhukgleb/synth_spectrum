import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from tsfit_utils import get_model_data
from typing import List, Union
from tsfit_utils import clean_pd
import scienceplots


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


def plot_metall(data: pd.DataFrame):
    metallicity = data["Fe_H"].to_numpy(float)
    error = data["chi_squared"].to_numpy(float)

    xy_point_density = np.vstack([metallicity, error])
    z_point_density = gaussian_kde(xy_point_density)(xy_point_density)
    idx_sort = z_point_density.argsort()
    x_plot, y_plot, z_plot = (
        metallicity[idx_sort],
        error[idx_sort],
        z_point_density[idx_sort],
    )

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
    ew = ew * 1000
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
        ax.set_xlim((0, 100))
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


def spectrum_converge(path2output: str):
    pass


def plot_metallVS(data_1: pd.DataFrame, data_2: pd.DataFrame):
    metallicity_1 = data_1["Fe_H"].to_numpy(float)
    error_1 = data_1["chi_squared"].to_numpy(float)

    xy_point_density_1 = np.vstack([metallicity_1, error_1])
    z_point_density_1 = gaussian_kde(xy_point_density_1)(xy_point_density_1)
    idx_sort_1 = z_point_density_1.argsort()
    x_plot_1, y_plot_1, z_plot_1 = (
        metallicity_1[idx_sort_1],
        error_1[idx_sort_1],
        z_point_density_1[idx_sort_1],
    )

    metallicity_2 = data_2["Fe_H"].to_numpy(float)
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
        _, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
        ax[0].set_title(r"Metallicity IRAS Z02229+6208")
        ax[0].set_ylabel(r"$\chi^{2}$")
        ax[0].set_xlabel(r"Metallicity, [Fe/H]")
        ax[0].set_ylim((0, 100))
        ax[1].set_title(r"Metallicity IRAS 07430+1115")
        ax[1].set_ylabel(r"$\chi^{2}$")
        ax[1].set_xlabel(r"Metallicity, [Fe/H]")
        ax[1].set_ylim((0, 100))
        density_1 = ax[0].scatter(x_plot_1, y_plot_1, c=z_plot_1)
        density_2 = ax[1].scatter(x_plot_2, y_plot_2, c=z_plot_2)

        plt.colorbar(density_2)
        plt.colorbar(density_1)

        plt.show()


if __name__ == "__main__":
    # t_path = "data/chem/02229_teff.dat"
    #     teff_graph(t_path)
    #
    from tsfit_utils import get_model_data
    from config_loader import tsfit_output

    out_1 = "2024-12-11-18-26-42_0.37586031193741987_LTE_Fe_1D"
    out_2 = "2024-12-10-15-46-05_0.926899482273021_LTE_Fe_1D"
    pd_data_1 = get_model_data(tsfit_output + out_1)
    pd_data_2 = get_model_data(tsfit_output + out_2)

    pd_data_1 = clean_pd(pd_data_1, True, True)
    pd_data_2 = clean_pd(pd_data_2, True, True)

    plot_metallVS(pd_data_1, pd_data_2)
    # plot_ion_balance(pd_data)
