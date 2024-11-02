import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
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


def plot_density_df_results(
    df_results: pd.DataFrame,
    x_axis_column: str,
    y_axis_column: str,
    xlim=None,
    ylim=None,
    invert_x_axis=False,
    invert_y_axis=False,
    **pltargs,
):
    if np.size(df_results[x_axis_column]) == 1:
        print("Only one point is found, so doing normal scatter plot")
        plot_scatter_df_results(
            df_results,
            x_axis_column,
            y_axis_column,
            xlim=xlim,
            ylim=ylim,
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
            **pltargs,
        )
        return
    try:
        with plt.style.context("science"):
            # creates density map for the plot
            x_array = df_results[x_axis_column]
            y_array = df_results[y_axis_column]
            xy_point_density = np.vstack([x_array, y_array])
            z_point_density = gaussian_kde(xy_point_density)(xy_point_density)
            idx_sort = z_point_density.argsort()
            x_plot, y_plot, z_plot = (
                x_array[idx_sort],
                y_array[idx_sort],
                z_point_density[idx_sort],
            )

            density = plt.scatter(
                x_plot, y_plot, c=z_plot, zorder=-1, vmin=0, **pltargs
            )

            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.colorbar(density)
            plt.xlabel(x_axis_column)
            plt.ylabel(y_axis_column)
            if invert_x_axis:
                plt.gca().invert_xaxis()
            if invert_y_axis:
                plt.gca().invert_yaxis()
            plt.show()
            plt.close()
    except LinAlgError:
        print("LinAlgError, so doing normal scatter plot")
        plot_scatter_df_results(
            df_results,
            x_axis_column,
            y_axis_column,
            xlim=xlim,
            ylim=ylim,
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
            **pltargs,
        )


if __name__ == "__main__":
    # t_path = "data/chem/02229_teff.dat"
    #     teff_graph(t_path)
    #
    from tsfit_utils import get_model_data
    from config_loader import tsfit_output

    out = "Oct-31-2024-23-28-47_0.2189189215158336_LTE_Fe_1D"
    pd_data = get_model_data(tsfit_output + out)
    print(pd_data)
