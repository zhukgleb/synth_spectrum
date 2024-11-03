from matplotlib.style import context
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import linregress, t
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

    res = linregress(rel_ew, metallicity)

    # ts = tinv(0.05, len(x) - 2)
    # print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
    # print(f"intercept (95%): {res.intercept:.6f}" f" +/- {ts*res.intercept_stderr:.6f}")
    print(f"Slope is: {res.slope}")
    with plt.style.context("science"):
        _, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(r"Ionization balance of IRAS Z02229+6208")
        ax.set_ylabel(r"Metallicity, [Fe/H]")
        ax.set_xlabel(r"$EW / \lambda$")
        ax.set_xlim((0, 100))
        plt.scatter(rel_ew, metallicity, color="black", alpha=0.5)
        plt.plot(rel_ew, res.intercept + res.slope * rel_ew, ls="--", color="black")
        plt.show()


if __name__ == "__main__":
    # t_path = "data/chem/02229_teff.dat"
    #     teff_graph(t_path)
    #
    from tsfit_utils import get_model_data
    from config_loader import tsfit_output

    out = "Oct-28-2024-16-30-31_0.8750247136632259_LTE_Fe_1D"
    pd_data = get_model_data(tsfit_output + out)
    plot_ion_balance(pd_data)
