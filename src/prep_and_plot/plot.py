"""
This module includes all necessary functions for the plotting functionality.
"""

import math
from copy import deepcopy

from matplotlib import colormaps
from matplotlib.colors import rgb2hex, LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
from os.path import join
from scipy import interpolate, optimize
from .data_helper import (
    get_polygon_from_json,
    load_demand,
    load_algorithm_results,
    save_algorithm_results,
)
from .plot_helper import *
from .setup_helper import create_default_paths, create_default_params


def plot_ba_cost(
    bpp: list,
    ba: list,
    cost: list,
    ba_comparisons: list[tuple[float, float]],
    cost_comparisons: list[tuple[float, float]],
    save_path: str,
    plot_cost: bool = True,
    ba_diff_zoom: bool = False,
    eco_opt_bpp: bool = False,
    ex_inf: bool = False,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    params: dict | None = None,
):
    """Plot bikeability and cost for algorithm and comparison point(s).

    Parameters
    ----------
    bpp : list
        Bike path percentage
    ba : list
        Bikeability
    cost : list
        Cost
    ba_comparisons : list[tuple[float, float]]
        Comparison points for bikeability [(bpp_comp, ba_comp)]
    cost_comparisons : list[tuple[float, float]]
        Comparison points for cost [(bpp_comp, cost_comp)]
    save_path : str
        Path to save the figure.
    plot_cost : bool
         Plot costs if true. (Default value = True)
    ba_diff_zoom : bool
         Plot zoomed in area of comparison points if true. (Default value = False)
    eco_opt_bpp : bool
         (Default value = False)
    ex_inf : bool
         (Default value = False)
    x_min : float
         Minimum x-axis (Default value = 0.0)
    x_max : float
         Maximum x-axis (Default value = 1.0)
    y_min : float
         Minimum y-axis (Default value = 0.0)
    y_max : float
         Maximum y-axis (Default value = 1.0)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()

    fig, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    ax1.set_xlabel(
        r"normalized relative length of bike paths $\lambda$",
        fontsize=params["fs_axl"],
    )

    ax1.plot(
        bpp,
        ba,
        c=params["c_ba"],
        label="bikeability",
        lw=params["lw_ba"],
    )
    bpp_vlines = dict()
    ba_hlines = dict()
    for idx, (bpp_comp, ba_comp) in enumerate(ba_comparisons):
        if idx == 0:
            ax1.plot(
                bpp_comp,
                ba_comp,
                c=params["c_ba"],
                ms=params["ms_ba"],
                marker=params["m_ba"],
            )
        bpp_x = min(bpp, key=lambda x: abs(x - bpp_comp))
        bpp_idx = next(x for x, val in enumerate(bpp) if val == bpp_x)
        ba_y = ba[bpp_idx]

        xmax, ymax = coord_transf(
            bpp_comp, max([ba_y, ba_comp]), xmax=1, xmin=0, ymax=1, ymin=0
        )
        if len(bpp_vlines) != 0:
            bpp_comp_close = min(bpp_vlines.keys(), key=lambda x: abs(x - bpp_comp))
            if abs(bpp_comp - bpp_comp_close) > 0.001:
                bpp_vlines[bpp_comp] = ymax
            else:
                bpp_vlines[bpp_comp_close] = max(bpp_vlines[bpp_comp_close], ymax)
        else:
            bpp_vlines[bpp_comp] = ymax

        if len(ba_hlines) != 0:
            ba_comp_close = min(ba_hlines.keys(), key=lambda x: abs(x - ba_comp))
            if abs(ba_comp - ba_comp_close) > 0.001:
                ba_hlines[ba_comp] = xmax
            else:
                ba_hlines[ba_comp_close] = max(ba_hlines[ba_comp_close], xmax)
        else:
            ba_hlines[ba_comp] = xmax
        ba_y_close = min(ba_hlines.keys(), key=lambda x: abs(x - ba_y))
        if abs(ba_y - ba_y_close) > 0.001:
            ba_hlines[ba_y] = xmax
        else:
            ba_hlines[ba_y_close] = max(ba_hlines[ba_y_close], xmax)

    for bpp_comp, ymax in bpp_vlines.items():
        ax1.axvline(
            x=bpp_comp,
            ymax=ymax,
            ymin=0,
            c=params["c_ba"],
            ls="--",
            alpha=0.5,
            lw=params["lw_ba"],
        )
    for ba_comp, xmax in ba_hlines.items():
        ax1.axhline(
            y=ba_comp,
            xmax=xmax,
            xmin=0,
            c=params["c_ba"],
            ls="--",
            alpha=0.5,
            lw=params["lw_ba"],
        )

    ax1.set_ylabel(
        r"bikeability b($\lambda$)",
        fontsize=params["fs_axl"],
        color=params["c_ba"],
    )
    ax1.tick_params(axis="y", labelsize=params["fs_ticks"], labelcolor=params["c_ba"])
    ax1.tick_params(axis="x", labelsize=params["fs_ticks"])
    if y_max - y_min < 2.0:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    if plot_cost:
        ax2 = ax1.twinx()
        for axis in ["top", "bottom", "left", "right"]:
            ax2.spines[axis].set_linewidth(0.5)
        ax2.set_ylim(y_min, y_max)

        ax2.plot(
            bpp,
            cost,
            c=params["c_cost"],
            label="total cost",
            lw=params["lw_cost"],
        )
        bpp_vlines = dict()
        cost_hlines = dict()
        for idx, (bpp_comp, cost_comp) in enumerate(cost_comparisons):
            if idx == 0:
                ax2.plot(
                    bpp_comp,
                    cost_comp,
                    c=params["c_cost"],
                    ms=params["ms_cost"],
                    marker=params["m_cost"],
                )
            bpp_x = min(bpp, key=lambda x: abs(x - bpp_comp))
            bpp_idx = next(x for x, val in enumerate(bpp) if val == bpp_x)
            cost_y = cost[bpp_idx]

            xmin, ymax = coord_transf(
                bpp_comp, cost_comp, xmax=1, xmin=0, ymax=1, ymin=0
            )

            if len(bpp_vlines) != 0:
                bpp_comp_close = min(bpp_vlines.keys(), key=lambda x: abs(x - bpp_comp))
                if abs(bpp_comp - bpp_comp_close) > 0.001:
                    bpp_vlines[bpp_comp] = ymax
                else:
                    bpp_vlines[bpp_comp_close] = max(bpp_vlines[bpp_comp_close], ymax)
            else:
                bpp_vlines[bpp_comp] = ymax

            if len(cost_hlines) != 0:
                cost_comp_close = min(
                    cost_hlines.keys(), key=lambda x: abs(x - cost_comp)
                )
                if abs(cost_comp - cost_comp_close) > 0.001:
                    cost_hlines[cost_comp] = xmin
                else:
                    cost_hlines[cost_comp_close] = min(
                        cost_hlines[cost_comp_close], xmin
                    )
            else:
                cost_hlines[cost_comp] = xmin
            cost_y_close = min(cost_hlines.keys(), key=lambda x: abs(x - cost_y))
            if abs(cost_y - cost_y_close) > 0.001:
                cost_hlines[cost_y] = xmin
            else:
                cost_hlines[cost_y_close] = min(cost_hlines[cost_y_close], xmin)

        for bpp_comp, ymax in bpp_vlines.items():
            ax2.axvline(
                x=bpp_comp,
                ymax=ymax,
                ymin=0,
                c=params["c_cost"],
                ls="--",
                alpha=0.5,
                lw=params["lw_cost"],
            )
        for cost_comp, xmin in cost_hlines.items():
            ax2.axhline(
                y=cost_comp,
                xmax=1,
                xmin=xmin,
                c=params["c_cost"],
                ls="--",
                alpha=0.5,
                lw=params["lw_cost"],
            )

        ax2.set_ylabel(
            "normalized cost", fontsize=params["fs_axl"], color=params["c_cost"]
        )
        ax2.tick_params(
            axis="y", labelsize=params["fs_ticks"], labelcolor=params["c_cost"]
        )

        if y_max - y_min < 2.0:
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    if params["titles"]:
        ax1.set_title("Bikeability and Cost", fontsize=params["fs_title"])

    if params["legends"]:
        handles = ax1.get_legend_handles_labels()[0]
        if plot_cost:
            handles.append(ax2.get_legend_handles_labels()[0][0])
        ax1.legend(
            handles=handles,
            loc="lower right",
            fontsize=params["fs_legend"],
            frameon=False,
        )

    if ba_diff_zoom:
        if bpp[0] <= 0.5:
            ax1ins = zoomed_inset_axes(ax1, 3.5, loc="upper right")
        else:
            ax1ins = zoomed_inset_axes(ax1, 3.5, bbox_to_anchor=(0.1, 0.975))

        mark_inset(ax1, ax1ins, loc1=2, loc2=3, fc="none", ec="0.7")

        bpp_comp = ba_comparisons[0][0]
        ba_comp = ba_comparisons[0][1]

        bpp_x = min(bpp, key=lambda x: abs(x - bpp_comp))
        bpp_idx = next(x for x, val in enumerate(bpp) if val == bpp_x)
        ba_y = ba[bpp_idx]

        x1 = round(bpp_comp - 0.05, 2)
        x2 = round(bpp_comp + 0.05, 2)
        y1 = round(min(ba_comp, ba_y) - 0.03, 2)
        y2 = min(round(max(ba_comp, ba_y) + 0.03, 2), 1)
        ax1ins.plot(bpp, ba, lw=params["lw_ba"])
        ax1ins.plot(
            bpp_comp,
            ba_comp,
            c=params["c_ba"],
            ms=params["ms_ba"],
            marker=params["m_ba"],
        )
        xmax, ymax = coord_transf(
            bpp_comp, max([ba_y, ba_comp]), xmin=x1, xmax=x2, ymin=y1, ymax=y2
        )
        ax1ins.axvline(
            x=bpp_comp, ymax=ymax, ymin=0, c=params["c_ba"], ls="--", alpha=0.5
        )
        ax1ins.axhline(
            y=ba_comp, xmax=xmax, xmin=0, c=params["c_ba"], ls="--", alpha=0.5
        )
        if abs(ba_y - ba_comp) > 0.0001:
            ax1ins.axhline(
                y=ba_y, xmax=xmax, xmin=0, c=params["c_ba"], ls="--", alpha=0.5
            )

        ax1ins.set_xlim(x1, x2)
        ax1ins.set_ylim(y1, y2)
        ax1ins.tick_params(
            axis="y", labelsize=params["fs_ticks"], labelcolor=params["c_ba"]
        )
        ax1ins.tick_params(axis="x", labelsize=params["fs_ticks"])
        ax1ins.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax1ins.xaxis.set_minor_locator(AutoMinorLocator())
        ax1ins.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax1ins.yaxis.set_minor_locator(AutoMinorLocator())

    if eco_opt_bpp:
        ax1_diff = ax1.twinx()
        ax1_diff.set_ylim(0.0, 1.0)
        ax1_diff.tick_params(
            axis="both",
            which="both",
            top=False,
            bottom=False,
            labelbottom=False,
            labeltop=False,
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )

        bpp_int, step = np.linspace(min(bpp), max(bpp), num=1000, retstep=True)
        if not ex_inf:
            f_ba = binding(bpp_int, *optimize.curve_fit(binding, bpp, ba)[0])
        else:
            f_ba = logistic(bpp_int, *optimize.curve_fit(logistic, bpp, ba)[0])
        ax1_diff.plot(bpp_int, f_ba, color="k")
        df_ba = np.array(
            [(f_ba[i + 1] - f_ba[i]) / step for i in range(len(bpp_int))[:-2]]
        )

        f_cost = interpolate.interp1d(bpp, cost)
        df_cost = np.array(
            [(f_cost(i + step) - f_cost(i)) / step for i in bpp_int[:-2]]
        )
        ba_cost_mes = df_ba - df_cost

        ax1ins_diff = inset_axes(
            ax1_diff, width="40%", height="40%", loc=4, borderpad=0.75
        )
        for axis in ["top", "bottom", "left", "right"]:
            ax1ins_diff.spines[axis].set_linewidth(0.5)
        ax1ins_diff.set_xlim(0.0, 1.0)
        ax1ins_diff.set_ylim(min(ba_cost_mes), max(ba_cost_mes))
        ax1ins_diff.plot(bpp_int[:-2], ba_cost_mes, c=params["c_ba"])
        ax1ins_diff.set_ylabel("", labelpad=1, fontsize=5)
        ax1ins_diff.tick_params(axis="y", length=2, width=0.5, pad=0.5, labelsize=4)
        ax1ins_diff.tick_params(axis="x", length=2, width=0.5, pad=0.5, labelsize=4)

        threshold = 0.0
        threshold_idx = next(x for x, val in enumerate(ba_cost_mes) if val < threshold)
        ax1ins_diff.axvline(
            x=bpp_int[threshold_idx],
            ymax=1,
            ymin=0,
            c="k",
            ls="--",
            alpha=0.5,
            lw=params["lw_cost"],
        )
        bpp_x = min(bpp, key=lambda x: abs(x - bpp_int[threshold_idx]))
        bpp_idx = next(x for x, val in enumerate(bpp) if val == bpp_x)
        ba_y = ba[bpp_idx]
        cost_y = cost[bpp_idx]
        ax1.axvline(
            x=bpp_x,
            ymax=ba_y,
            ymin=0,
            c=params["c_ba"],
            ls="--",
            alpha=0.5,
            lw=params["lw_ba"],
        )
        ax1.axhline(
            y=ba_y,
            xmax=bpp_x,
            xmin=0,
            c=params["c_ba"],
            ls="--",
            alpha=0.5,
            lw=params["lw_ba"],
        )
        ax2.axvline(
            x=bpp_x,
            ymax=cost_y,
            ymin=0,
            c=params["c_cost"],
            ls="--",
            alpha=0.5,
            lw=params["lw_cost"],
        )
        ax2.axhline(
            y=cost_y,
            xmax=1,
            xmin=bpp_x,
            c=params["c_cost"],
            ls="--",
            alpha=0.5,
            lw=params["lw_cost"],
        )

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_los_nos(
    bpp: list,
    los: list,
    nos: list,
    los_comparisons: list[tuple[float, float]],
    nos_comparisons: list[tuple[float, float]],
    save_path: str,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    params: dict | None = None,
):
    """

    Parameters
    ----------
    bpp : list
        Bike path percentage
    los : list
        Length on street without a bike path.
    nos : list
        Number of cyclists on streets without a bike path.
    los_comparisons : list[tuple[float, float]]
        Comparison points for LoS
    nos_comparisons : list[tuple[float, float]]
        Comparison points for NoS
    save_path : str
        Path to save the figure.
    x_min : float
         Minimum x-axis (Default value = 0.0)
    x_max : float
         Maximum x-axis (Default value = 1.0)
    y_min : float
         Minimum y-axis (Default value = 0.0)
    y_max : float
         Maximum y-axis (Default value = 1.0)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()

    fig, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_los_nos"])
    ax2 = ax1.twinx()
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
        ax2.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.set_xlabel(
        r"normalized relative length of bike paths $\lambda$",
        fontsize=params["fs_axl"],
    )

    (p1,) = ax1.plot(
        bpp,
        los,
        label="length",
        c=params["c_los"],
        lw=params["lw_los"],
    )

    bpp_vlines = dict()
    los_hlines = dict()
    for idx, (bpp_comp, los_comp) in enumerate(los_comparisons):
        if idx == 0:
            ax1.plot(
                bpp_comp,
                los_comp,
                c=params["c_los"],
                ms=params["ms_los"],
                marker=params["m_los"],
            )

        bpp_x = min(bpp, key=lambda x: abs(x - bpp_comp))
        bpp_idx = next(x for x, val in enumerate(bpp) if val == bpp_x)
        los_y = los[bpp_idx]

        xmax, ymax = coord_transf(bpp_comp, los_comp, xmax=1, xmin=0, ymax=1, ymin=0)
        if len(bpp_vlines) != 0:
            bpp_comp_close = min(bpp_vlines.keys(), key=lambda x: abs(x - bpp_comp))
            if abs(bpp_comp - bpp_comp_close) > 0.001:
                bpp_vlines[bpp_comp] = ymax
            else:
                bpp_vlines[bpp_comp_close] = max(bpp_vlines[bpp_comp_close], ymax)
        else:
            bpp_vlines[bpp_comp] = ymax

        if len(los_hlines) != 0:
            los_comp_close = min(los_hlines.keys(), key=lambda x: abs(x - los_comp))
            if abs(los_comp - los_comp_close) > 0.001:
                los_hlines[los_comp] = xmax
            else:
                los_hlines[los_comp_close] = max(los_hlines[los_comp_close], xmax)
        else:
            los_hlines[los_comp] = xmax
        los_y_close = min(los_hlines.keys(), key=lambda x: abs(x - los_y))
        if abs(los_y - los_y_close) > 0.001:
            los_hlines[los_y] = xmax
        else:
            los_hlines[los_y_close] = max(los_hlines[los_y_close], xmax)

    for bpp_comp, ymax in bpp_vlines.items():
        ax1.axvline(
            x=bpp_comp,
            ymax=ymax,
            ymin=0,
            c=params["c_los"],
            ls="--",
            alpha=0.5,
            lw=params["lw_los"],
        )
    for los_comp, xmax in los_hlines.items():
        ax1.axhline(
            y=los_comp,
            xmax=xmax,
            xmin=0,
            c=params["c_los"],
            ls="--",
            alpha=0.5,
            lw=params["lw_los"],
        )

    ax1.set_ylabel("length on street", fontsize=params["fs_axl"], color=params["c_los"])
    ax1.tick_params(axis="y", labelsize=params["fs_ticks"], labelcolor=params["c_los"])
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    (p2,) = ax2.plot(
        bpp,
        nos,
        label="cyclists",
        c=params["c_nos"],
        lw=params["lw_nos"],
    )

    bpp_vlines = dict()
    nos_hlines = dict()
    for idx, (bpp_comp, nos_comp) in enumerate(nos_comparisons):
        if idx == 0:
            ax2.plot(
                bpp_comp,
                nos_comp,
                c=params["c_nos"],
                ms=params["ms_nos"],
                marker=params["m_nos"],
            )

        bpp_x = min(bpp, key=lambda x: abs(x - bpp_comp))
        bpp_idx = next(x for x, val in enumerate(bpp) if val == bpp_x)
        nos_y = nos[bpp_idx]

        xmin, ymax = coord_transf(bpp_comp, nos_comp, xmax=1, xmin=0, ymax=1, ymin=0)

        if len(bpp_vlines) != 0:
            bpp_comp_close = min(bpp_vlines.keys(), key=lambda x: abs(x - bpp_comp))
            if abs(bpp_comp - bpp_comp_close) > 0.001:
                bpp_vlines[bpp_comp] = ymax
            else:
                bpp_vlines[bpp_comp_close] = max(bpp_vlines[bpp_comp_close], ymax)
        else:
            bpp_vlines[bpp_comp] = ymax

        if len(nos_hlines) != 0:
            nos_comp_close = min(nos_hlines.keys(), key=lambda x: abs(x - cost_comp))
            if abs(nos_comp - nos_comp_close) > 0.001:
                nos_hlines[nos_comp] = xmin
            else:
                nos_hlines[nos_comp_close] = min(nos_hlines[nos_comp_close], xmin)
        else:
            nos_hlines[nos_comp] = xmin
        nos_y_close = min(nos_hlines.keys(), key=lambda x: abs(x - nos_y))
        if abs(nos_y - nos_y_close) > 0.001:
            nos_hlines[nos_y] = xmin
        else:
            nos_hlines[nos_y_close] = min(nos_hlines[nos_y_close], xmin)

    for bpp_comp, ymax in bpp_vlines.items():
        ax2.axvline(
            x=bpp_comp,
            ymax=ymax,
            ymin=0,
            c=params["c_nos"],
            ls="--",
            alpha=0.5,
            lw=params["lw_nos"],
        )
    for cost_comp, xmin in nos_hlines.items():
        ax2.axhline(
            y=cost_comp,
            xmax=1,
            xmin=xmin,
            c=params["c_nos"],
            ls="--",
            alpha=0.5,
            lw=params["lw_nos"],
        )
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.set_ylabel(
        "cyclists on street", fontsize=params["fs_axl"], color=params["c_nos"]
    )
    ax2.tick_params(axis="y", labelsize=params["fs_ticks"], labelcolor=params["c_nos"])

    if params["titles"]:
        ax1.set_title("Cyclists and Length on Street", fontsize=params["fs_title"])
    if params["legends"]:
        ax1.legend(
            [p1, p2],
            [l.get_label() for l in [p1, p2]],
            fontsize=params["fs_legend"],
            frameon=False,
        )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_street_network(
    city: str,
    save: str,
    G: nx.MultiGraph | nx.MultiDiGraph,
    plot_folder: str,
    params: dict | None = None,
):
    """Plots the street network of graph G.

    Parameters
    ----------
    city : str
        Name of the city/area
    save : str
        Save name of the city/area
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    plot_folder : str
        Path of the folder to save the plots.
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()

    fig, ax = plt.subplots(figsize=params["figs_snetwork"], dpi=params["dpi"])
    plot_graph(
        G,
        ax=ax,
        bgcolor="#ffffff",
        show=False,
        close=False,
        node_color=params["nc_snetwork"],
        node_size=params["ns_snetwork"],
        node_zorder=3,
        edge_color=params["ec_snetwork"],
        edge_linewidth=params["ew_snetwork"],
    )
    if params["titles"]:
        fig.suptitle(f"Graph used for {city.capitalize()}", fontsize=params["fs_title"])

    plt.savefig(join(plot_folder, f'{save}_street_network.{params["plot_format"]}'))


def plot_used_nodes(
    city: str,
    save: str,
    G: nx.MultiGraph | nx.MultiDiGraph,
    trip_nbrs: dict,
    stations: list,
    plot_folder: str,
    params: dict | None = None,
):
    """Plots usage of nodes in graph G.

    Parameters
    ----------
    city : str
        Name of the city/area
    save : str
        Save name of the city/area
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    trip_nbrs : dict
        Demand, hould be structured as returned from load_trips().
    stations : list
        Nodes used as stations, should be structured as returned from load_trips().
    plot_folder : str
        Path of the folder to save the plots.
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    print("Plotting used nodes.")
    if params is None:
        params = create_default_params()

    node_load = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if e_node == s_node:
                continue
            if (s_node, e_node) in trip_nbrs:
                node_load[s_node] += sum(trip_nbrs[(s_node, e_node)].values())
                node_load[e_node] += sum(trip_nbrs[(s_node, e_node)].values())

    node_load = {n: int(t / params["stat_usage_norm"]) for n, t in node_load.items()}

    max_load = max(node_load.values())  # 191821
    print(f"Maximal station usage: {max_load}")
    min_load = min([load for n, load in node_load.items() if n in stations])
    print(f"Minimal station usage: {min_load}")

    r = magnitude(max_load)

    hist_data = [load for n, load in node_load.items() if n in stations]
    hist_save = join(
        plot_folder, f'{save}_stations_usage_distribution.{params["plot_format"]}'
    )
    hist_xlim = (0.0, round(max_load, -(r - 1)))

    cmap = colormaps[params["cmap_nodes"]]

    plot_histogram(
        hist_data,
        hist_save,
        xlabel="total number of trips per year",
        ylabel="number of stations",
        xlim=hist_xlim,
        bins=25,
        cm=cmap,
        dpi=params["dpi"],
        figsize=params["figs_station_usage_hist"],
    )

    vmin = min_load
    divnorm = LogNorm(vmin=min_load, vmax=max_load, clip=True)
    node_size = [params["nodesize"] if n in stations else 0 for n in G.nodes()]

    node_color = [rgb2hex(cmap(divnorm(node_load[n]))) for n in G.nodes()]

    fig2, ax2 = plt.subplots(dpi=params["dpi"], figsize=params["figs_station_usage"])
    plot_graph(
        G,
        ax=ax2,
        bgcolor="#ffffff",
        node_size=node_size,
        node_color=node_color,
        node_zorder=3,
        edge_color=params["ec_station_usage"],
        edge_linewidth=params["ew_snetwork"],
        show=False,
        close=False,
    )

    sm = plt.cm.ScalarMappable(
        cmap=colormaps[params["cmap_nodes"]],
        norm=divnorm,
    )

    # cbaxes = fig2.add_axes([0.1, 0.05, 0.8, 0.03])
    divider = make_axes_locatable(ax2)
    cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)

    max_r = magnitude(max_load)
    min_r = magnitude(vmin)

    if vmin < 10:
        f_tick = vmin
    else:
        f_tick = int(round(vmin, min(-1, -(min_r - 2))))
    if round(max_load / 2) < 10:
        m_tick = int(round((max_load - vmin) / 2))
    else:
        m_tick = int(round((max_load - vmin) / 2, min(-1, -(max_r - 2))))
    if max_load < 10:
        l_tick = max_load
    else:
        l_tick = int(round(max_load, min(-1, -(max_r - 2))))

    cbar = fig2.colorbar(
        sm,
        ax=ax2,
        orientation="horizontal",
        cax=cbaxes,
        ticks=[f_tick, m_tick, l_tick],
    )
    cbar.ax.set_xticklabels([f_tick, m_tick, l_tick])

    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(axis="y", labelsize=params["fs_ticks"], width=0.5)
    cbar.ax.set_ylabel("total number of trips", fontsize=params["fs_axl"])

    if params["titles"]:
        fig2.suptitle(f"{city.capitalize()}", fontsize=params["fs_title"])

    fig2.savefig(
        join(plot_folder, f'{save}_stations_used.{params["plot_format"]}'),
        bbox_inches="tight",
    )

    plt.close("all")


def plot_edge_load(
    G: nx.MultiGraph | nx.MultiDiGraph,
    edge_load: dict,
    save: str,
    buildup: bool,
    minmode: int,
    ex_inf: bool,
    data_set: str,
    plot_folder: str,
    edge_color: str | Iterable[str] | None = None,
    edge_width: float | Sequence[float] = 1,
    edge_alpha: float | None = None,
    edge_zorder: int | Iterable[int] = 1,
    legend_colors: list | None = None,
    legend_titles: list | None = None,
    params: dict | None = None,
):
    """Plotting the edge load of the network for a given step.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    edge_load : dict
        Edg load during the algorithm. key: edge, value: list with load per step
    save : str
        Save name of the city/area
    buildup : bool
        If False normal reversed mode is assumed, if True building from current state.
    minmode : int
        Which minmode for edge choosing was used.
    ex_inf : bool
        If existing infrastructure should be considered.
    data_set : str
        For which data set the plot is done for e.g. 'cs'.
    plot_folder: str
        Folder to save the plot
    edge_color : list | None
        Color of the edges. (Default value = None)
    edge_width :
        Width of the edges. (Default value = None)
    edge_alpha : float
        Opacity of the edges.
    edge_zorder
        The zorder to plot edges. If single integer all edges have the same zorder.
    legend_colors :
         (Default value = None)
    legend_titles :
         (Default value = None)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()

    nx.set_edge_attributes(G, 0, "load")

    edge_load = {(k[0], k[1], 0): v for k, v in edge_load.items()}
    nx.set_edge_attributes(G, 0, "load")
    nx.set_edge_attributes(G, edge_load, "load")

    fig, ax = plt.subplots(dpi=600, figsize=params["figs_bp_comp"])
    ax.set_facecolor("#ffffff")
    if edge_color is None:
        edge_color = ox.plot.get_edge_colors_by_attr(G, "load", cmap="viridis")
    if edge_width is None:
        edge_width = np.array(
            [0 if d["load"] == 0 else 1 for u, v, d in G.edges(data=True)]
        )
    if edge_zorder is None:
        min_non_zero_load = min([v for v in edge_load.values() if abs(v) > 0.0])
        edge_zorder = np.array([d["load"] for u, v, d in G.edges(data=True)])
        edge_zorder[edge_zorder == 0] = 0.9 * min_non_zero_load
        edge_zorder = 1 / edge_zorder

    plot_graph(
        G,
        bgcolor="#ffffff",
        ax=ax,
        node_size=0,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        edge_zorder=edge_zorder,
        edge_alpha=edge_alpha,
        show=False,
        close=False,
    )

    if legend_colors is not None:
        lw_leg = params["lw_legend_bp_evo"]
        for idx, leg_colors in enumerate(legend_colors):
            bbox = (idx / len(legend_colors) + 0.1, -0.05, 0.5, 1)
            leg_bars = [
                Line2D([0], [0], color=color_i, lw=lw_leg)
                for color_i in leg_colors.values()
            ]
            leg_name = [name_i for name_i in legend_colors[0].keys()]
            legend = ax.legend(
                leg_bars,
                leg_name,
                title=legend_titles[idx],
                bbox_to_anchor=bbox,
                loc=3,
                ncol=2,
                mode=None,
                borderaxespad=0.0,
                fontsize=params["fs_legend"],
                frameon=False,
            )
            ax.add_artist(legend)

        ax.legend(
            [
                Line2D([0], [0], color="k", alpha=0, lw=0)
                for color_i in legend_colors[0].values()
            ],
            ["" for name_i in legend_colors[0].keys()],
            title="",
            bbox_to_anchor=(0, -0.05, 0.5, 1),
            loc=3,
            ncol=2,
            mode=None,
            borderaxespad=0.0,
            fontsize=params["fs_legend"],
            frameon=False,
        )

    plt.savefig(
        join(
            plot_folder,
            f'{save}-load-{buildup:d}{minmode}{ex_inf:d}_{data_set}.'
            f'{params["plot_format"]}',
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_bp_comparison(
    city: str,
    save: str,
    G: nx.MultiGraph | nx.MultiDiGraph,
    ee_algo: list,
    ee_cs: list | None,
    bpp_algo: list,
    bpp_cs: list,
    buildup: bool,
    minmode: int,
    ex_inf: bool,
    plot_folder: str,
    save_name: str | None = None,
    mode="diff",
    edge_loads_algo: dict | None = None,
    edge_load_cs: dict | None = None,
    idx: int | None = None,
    params: dict | None = None,
):
    """

    Parameters
    ----------
    city : str
        Name of the city/area
    save : str
        Save name of the city/area
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    ee_algo : list
        Order of edited edges for the full run.
    ee_cs : list | None
        Edited edges for the comparison point.
    bpp_algo : list
        Bike path percentage for the full run.
    bpp_cs : list
        Bike path percentage for the comparison point.
    buildup : bool
        If False normal reversed mode is assumed, if True building from current state.
    minmode : int
        Which minmode for edge choosing was used.
    ex_inf : bool
        If existing infrastructure should be considered.
    plot_folder : str
        Path of the folder to save the plots.
    save_name : str
         (Default value = None)
    mode : str
         (Default value = "diff")
    edge_loads_algo : dict | None
         (Default value = None)
    edge_load_cs : dict | None
         (Default value = None)
    idx : int | None
         (Default value = None)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------


    """
    if edge_load_cs is None:
        edge_load_cs = dict()
    if edge_loads_algo is None:
        edge_loads_algo = dict()
    nx.set_edge_attributes(G, False, "algo")
    nx.set_edge_attributes(G, False, "cs")

    if buildup:
        ee_algo = ee_algo
        bpp_algo = bpp_algo
        edge_loads_algo = edge_loads_algo
    else:
        ee_algo = list(reversed(ee_algo))
        bpp_algo = list(reversed(bpp_algo))
        if set(map(type, edge_loads_algo.values())) == {list}:
            edge_loads_algo = {k: list(reversed(v)) for k, v in edge_loads_algo.items()}

    if idx is None:
        idx = min(range(len(bpp_algo)), key=lambda i: abs(bpp_algo[i] - bpp_cs))

    if set(map(type, edge_loads_algo.values())) == {list}:
        edge_loads_algo_plot = {k: v[idx] for k, v in edge_loads_algo.items()}
    else:
        edge_loads_algo_plot = edge_loads_algo

    plot_edge_load(
        G,
        edge_loads_algo_plot,
        save,
        buildup,
        minmode,
        ex_inf,
        "algo",
        plot_folder,
        params=params,
    )
    plot_edge_load(
        G,
        edge_load_cs,
        save,
        buildup,
        minmode,
        ex_inf,
        "cs",
        plot_folder,
        params=params,
    )

    print(
        f"Difference in BPP between cs and algo: " f"{abs(bpp_cs - bpp_algo[idx]):4.3f}"
    )

    ee_algo_cut = ee_algo[:idx]
    for edge in ee_algo_cut:
        G[edge[0]][edge[1]][0]["algo"] = True
    if ee_cs is not None:
        for edge in ee_cs:
            G[edge[0]][edge[1]][0]["cs"] = True

    ec = []
    edge_zorder = []

    len_algo = 0
    len_cs = 0
    len_both = 0

    for u, v, data in G.edges(data=True):
        if not isinstance(data["ex_inf"], bool):
            data["ex_inf"] = literal_eval(data["ex_inf"])

    if mode == "algo":
        for u, v, k, data in G.edges(keys=True, data=True):
            if data["algo"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_algo"])
                    edge_zorder.append(0)
            else:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_unused"])
                    edge_zorder.append(4)
    elif mode == "cs":
        for u, v, k, data in G.edges(keys=True, data=True):
            if data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_cs"])
                    edge_zorder.append(1)
            else:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_unused"])
                    edge_zorder.append(4)
    elif mode == "diff":
        for u, v, k, data in G.edges(keys=True, data=True):
            if data["algo"] and data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_both_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_both"])
                    edge_zorder.append(2)
                len_both += data["length"]
            elif data["algo"] and not data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_algo"])
                    edge_zorder.append(0)
                len_algo += data["length"]
            elif not data["algo"] and data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_cs"])
                    edge_zorder.append(1)
                len_cs += data["length"]
            else:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_unused"])
                    edge_zorder.append(4)
        print(
            f"Overlap between p+s and algo: "
            f"{len_both / (len_cs + len_both + len_algo):3.2f}"
        )
    elif mode == "diff_ex_inf":
        for u, v, k, data in G.edges(keys=True, data=True):
            if data["algo"] and data["cs"]:
                if data["ex_inf"]:
                    ec.append(params["color_both_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_both"])
                    edge_zorder.append(2)
                len_both += data["length"]
            elif data["algo"] and not data["cs"]:
                if data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_algo"])
                    edge_zorder.append(0)
                len_algo += data["length"]
            elif not data["algo"] and data["cs"]:
                if data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_cs"])
                    edge_zorder.append(1)
                len_cs += data["length"]
            else:
                if data["ex_inf"]:
                    ec.append(params["color_both_ex_inf"])
                    edge_zorder.append(3)
                else:
                    ec.append(params["color_unused"])
                    edge_zorder.append(4)
        print(
            f"Overlap between p+s and algo: "
            f"{len_both / (len_cs + len_both + len_algo):3.2f}"
        )
    elif mode == "ex_inf":
        for u, v, k, data in G.edges(keys=True, data=True):
            if data["ex_inf"]:
                ec.append(params["color_ex_inf"])
                edge_zorder.append(3)
            else:
                ec.append(params["color_unused"])
                edge_zorder.append(4)
    else:
        print("You have to choose between algo, p+s and diff.")

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_bp_comp"])
    plot_graph(
        G,
        bgcolor="#ffffff",
        ax=ax,
        node_size=0,
        node_color=params["nc_bp_comp"],
        node_zorder=3,
        edge_linewidth=params["ew_snetwork"],
        edge_color=ec,
        edge_zorder=edge_zorder,
        show=False,
        close=False,
    )

    if params["legends"]:
        lw_leg = params["lw_legend_bp_evo"]
        if mode == "algo":
            leg = [
                Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                Line2D([0], [0], color=params["color_unused"], lw=lw_leg),
            ]
            ax.legend(
                leg,
                ["Algorithm", "None"],
                bbox_to_anchor=(0, -0.05, 1, 1),
                loc=3,
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
                fontsize=params["fs_legend"],
                frameon=False,
            )
        elif mode == "cs":
            leg = [
                Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                Line2D([0], [0], color=params["color_unused"], lw=lw_leg),
            ]
            ax.legend(
                leg,
                ["Primary + Secondary", "None"],
                bbox_to_anchor=(0, -0.05, 1, 1),
                loc=3,
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
                fontsize=params["fs_legend"],
                frameon=False,
            )
        elif mode == "diff" and not ex_inf:
            leg = [
                Line2D([0], [0], color=params["color_both"], lw=lw_leg),
                Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                Line2D([0], [0], color=params["color_unused"], lw=lw_leg),
            ]
            ax.legend(
                leg,
                ["Both", "Algorithm", "Primary+Secondary", "None"],
                bbox_to_anchor=(0, -0.05, 1, 1),
                loc=3,
                ncol=4,
                mode="expand",
                borderaxespad=0.0,
                fontsize=params["fs_legend"],
                frameon=False,
            )
        elif (mode == "diff" and ex_inf) or mode == "diff_ex_inf":
            leg = [
                Line2D([0], [0], color=params["color_both"], lw=lw_leg),
                Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                Line2D([0], [0], color=params["color_both_ex_inf"], lw=lw_leg),
                Line2D([0], [0], color=params["color_unused"], lw=lw_leg),
            ]
            ax.legend(
                leg,
                [
                    "Both",
                    "Algorithm",
                    "Primary+Secondary",
                    "Existing Bike Paths",
                    "None",
                ],
                bbox_to_anchor=(0, -0.05, 1, 1),
                loc=3,
                ncol=3,
                mode="expand",
                borderaxespad=0.0,
                fontsize=params["fs_legend"],
                frameon=False,
            )
        elif mode == "ex_inf":
            leg = [
                Line2D([0], [0], color=params["color_ex_inf"], lw=lw_leg),
                Line2D([0], [0], color=params["color_unused"], lw=lw_leg),
            ]
            ax.legend(
                leg,
                ["Ex Inf", "None"],
                bbox_to_anchor=(0, -0.05, 1, 1),
                loc=3,
                ncol=2,
                mode="expand",
                borderaxespad=0.0,
                fontsize=params["fs_legend"],
                frameon=False,
            )

    if params["titles"]:
        if mode == "algo":
            ax.set_title(f"{city.capitalize()}: Algorithm", fontsize=params["fs_title"])
        elif mode == "cs":
            ax.set_title(
                f"{city.capitalize()}: Primary/Secondary", fontsize=params["fs_title"]
            )
        elif mode == "diff" and not ex_inf:
            ax.set_title(f"{city}: Comparison", fontsize=params["fs_title"])
        elif (mode == "diff" and ex_inf) or mode == "diff_ex_inf":
            ax.set_title(f"{city}: Comparison with ex inf", fontsize=params["fs_title"])
        elif mode == "ex_inf":
            ax.set_title(f"{city}: Ex Inf", fontsize=params["fs_title"])

    if save_name is None:
        save_name = f"{save}-bp-build-{buildup:d}{minmode}{ex_inf:d}_{mode}"

    plt.savefig(
        join(plot_folder, f'{save_name}.{params["plot_format"]}'),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_edges(
    G: nx.MultiGraph | nx.MultiDiGraph,
    edge_color: str | Iterable[str],
    node_size: float | Iterable[float],
    save_path: str,
    title: str = "",
    figsize: tuple[float, float] = (12, 12),
    params: dict | None = None,
):
    """

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    edge_color : Iterable
        Color of the edges
    node_size : float | Iterable[float],
        Size of the nodes
    save_path : str
        Path of the plot file.
    title : str
         Title of the figure (Default value = "")
    figsize : tuple[float, float]
         Size of the figure (Default value = (12, 12)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    fig, ax = plt.subplots(dpi=params["dpi"], figsize=figsize)
    plot_graph(
        G,
        ax=ax,
        bgcolor="#ffffff",
        edge_color=edge_color,
        edge_linewidth=params["ew_snetwork"],
        node_color=params["nc_bp_comp"],
        node_size=node_size,
        node_zorder=3,
        show=False,
        close=False,
    )

    if params["titles"]:
        ax.set_title(title, fontsize=params["fs_axl"])
    if params["legends"]:
        leg = [
            Line2D([0], [0], color=params["color_algo"], lw=params["lw_legend_bp_evo"]),
            Line2D(
                [0], [0], color=params["color_ex_inf"], lw=params["lw_legend_bp_evo"]
            ),
        ]
        ax.legend(
            leg,
            ["Algorithm", "Existing Bike Paths"],
            bbox_to_anchor=(0, -0.05, 1, 1),
            loc=3,
            ncol=2,
            mode="expand",
            borderaxespad=0.0,
            fontsize=params["fs_legend"],
            frameon=False,
        )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_bp_evo(
    save: str,
    G: nx.MultiGraph | nx.MultiDiGraph,
    edited_edges: list,
    bpp: list,
    buildup: bool,
    minmode: int,
    ex_inf: bool,
    plot_folder: str,
    params: dict | None = None,
):
    """

    Parameters
    ----------
    save : str
        Save name of the city/area
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    edited_edges : list
        Order of edges edited
    bpp : list
        Bike path percentage
    buildup : bool
        If False normal reversed mode is assumed, if True building from current state.
    minmode : int
        Which minmode for edge choosing was used.
    ex_inf : bool
        If existing infrastructure should be considered.
    plot_folder : str
        Path of the folder to save the plots.
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------


    """
    print(f"Begin plotting bike path evolution.")

    evo_params = deepcopy(params)
    evo_params["titles"] = True

    plot_folder_evo = join(plot_folder, "evolution")
    Path(plot_folder_evo).mkdir(parents=True, exist_ok=True)
    base_save = join(
        plot_folder_evo, f"{save}-edited-mode-{buildup:d}{minmode}{ex_inf:d}-"
    )

    if buildup:
        ee = edited_edges
    else:
        ee = list(reversed(edited_edges))

    for i in evo_params["bpp_range"]:
        idx = next(
            x for x, val in enumerate(bpp) if val == min(bpp, key=lambda x: abs(x - i))
        )
        ee_evo = ee[:idx]
        ec_evo = get_edge_color_bp(
            G,
            ee_evo,
            evo_params["color_algo"],
            ex_inf,
            evo_params["color_algo_ex_inf"],
            evo_params["color_unused"],
        )
        fig_nbr = f"{i*100:4.2f}".replace(".", "_")
        save_path = f"{base_save}{fig_nbr}.png"
        plot_edges(
            G,
            ec_evo,
            0,
            save_path,
            title=f"normalized relative length of bike paths: {i:4.3f}",
            figsize=evo_params["figs_bp_comp"],
            params=evo_params,
        )
        if (i * 100) % 10 == 0:
            print(f"Reached {i:4.3f} bpp.")

    print("Finished plotting bike path evolution.")


def plot_edge_load_evo(
    save: str,
    G: nx.MultiGraph | nx.MultiDiGraph,
    edge_loads: list,
    bpp: list,
    node_size: float | Iterable[float],
    buildup: bool,
    minmode: int,
    ex_inf: bool,
    plot_folder: str,
    params: dict | None = None,
):
    """

    Parameters
    ----------
    save : str
        Save name of the city/area
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    edge_loads : list
        Load of the edges
    bpp : list
        Bike path percentage
    node_size : float | Iterable[float]
        Size of the nodes
    buildup : bool
        If False normal reversed mode is assumed, if True building from current state.
    minmode : int
        Which minmode for edge choosing was used.
    ex_inf : bool
        If existing infrastructure should be considered.
    plot_folder : str
        Path of the folder to save the plots.
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------


    """
    print("Begin plotting load evolution.")

    plot_folder_evo = join(plot_folder, "evolution")
    Path(plot_folder_evo).mkdir(parents=True, exist_ok=True)

    if buildup:
        edge_loads = edge_loads
    else:
        edge_loads = {k: list(reversed(v)) for k, v in edge_loads.items()}

    max_load = max([max([v for v in edge_load]) for edge_load in edge_loads.values()])

    plots = range(0, int(max(bpp) * 100) + 1)
    for i in plots:
        idx = next(
            x
            for x, val in enumerate(bpp)
            if val == min(bpp, key=lambda x: abs(x - i / 100))
        )

        nx.set_edge_attributes(G, 0, "load")
        edge_load_evo = {k: v[idx] for k, v in edge_loads.items()}
        edge_load_evo = {literal_eval(k): v for k, v in edge_load_evo.items()}
        edge_load_evo = {
            (k[0], k[1], 0): v / max_load for k, v in edge_load_evo.items()
        }
        nx.set_edge_attributes(G, edge_load_evo, "load")
        ec_evo = ox.plot.get_edge_colors_by_attr(G, "load", cmap="magma_r")

        save_path = join(
            plot_folder_evo,
            f'{save}-load-mode-{buildup:d}{minmode}{ex_inf:d}-{i}.'
            f'{params["plot_format"]}',
        )
        if params["titles"]:
            plot_edges(
                G,
                ec_evo,
                node_size,
                save_path,
                title=f"Fraction of Bike Paths: {i}%",
                figsize=params["figs_bp_comp"],
                params=params,
            )
        else:
            plot_edges(
                G,
                ec_evo,
                node_size,
                save_path,
                figsize=params["figs_bp_comp"],
                params=params,
            )
    print("Finished plotting load evolution.")


def plot_mode(
    city: str,
    save: str,
    data: dict,
    data_cs: dict,
    data_base: dict,
    data_opt: dict,
    data_ex_inf: dict,
    G: nx.MultiGraph | nx.MultiDiGraph,
    mode: tuple[bool, int, bool],
    end_plot: int,
    evaluation_data: dict,
    plot_folder: str,
    evo: bool = False,
    params: dict | None = None,
    paths: dict | None = None,
):
    """Plots the results for one mode for the given city. If no 'params' are given
    the default ones are created and used.

    Parameters
    ----------
    city : str
        Name of the city/area
    save : str
        Save name of the city/area
    data : dict
        Results from the complete run
    data_cs : dict
        Results from a comparison point
    data_base : dict
        Results for the base case
    data_opt : dict
        Results for the optimal case
    data_ex_inf :
        Results for just existing infrastructure
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    mode : tuple[bool, int, bool]
        Mode of the complete run (buildup, minmode, ex inf)
    end_plot : int
        Index for last data point used for plotting.
    evaluation_data: dict
        Storage to save the evaluation data to.
    plot_folder : str
        Path of the folder to save the plots.
    evo : bool
        If true plot evolution of bike path network. (Default value = False)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict | None
        Dict with the paths for data, plots etc., check 'setup_paths.py' in the
        'scripts' folder. (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()

    buildup = mode[0]
    minmode = mode[1]
    ex_inf = mode[2]

    bike_paths_cs = data_cs["bike_paths"]
    bike_path_perc_cs = data_cs["bike_path_percentage"]
    cost_cs = data_cs["total_cost"]
    total_real_distance_traveled_cs = data_cs["total_real_distances_traveled"]
    total_felt_distance_traveled_cs = data_cs["total_felt_distance_traveled"]
    nos_cs = data_cs["number_on_street"]

    bike_path_perc_ex_inf = data_ex_inf["bike_path_percentage"]
    cost_ex_inf = data_ex_inf["total_cost"]
    total_felt_distance_traveled_ex_inf = data_ex_inf["total_felt_distance_traveled"]

    total_real_distance_traveled_base = data_base["total_real_distances_traveled"]
    total_felt_distance_traveled_base = data_base["total_felt_distance_traveled"]

    total_real_distance_traveled_opt = data_opt["total_real_distances_traveled"]
    total_felt_distance_traveled_opt = data_opt["total_felt_distance_traveled"]

    edited_edges = data["edited_edges"]
    bike_path_perc = data["bike_path_percentage"]
    cost = data["total_cost"]
    total_real_distance_traveled = data["total_real_distances_traveled"]
    total_felt_distance_traveled = data["total_felt_distance_traveled"]
    nbr_on_street = data["number_on_street"]
    edge_loads = load_comparison_state_results(
        join(
            paths["output_folder"],
            save,
            f"{save}_data_algo_comp_edge_load_{buildup:d}{minmode}{ex_inf:d}.json",
        )
    )["edge_loads"]

    _, trdt_base = total_distance_traveled_list(
        total_real_distance_traveled, total_real_distance_traveled_base, buildup
    )
    tfdt_base = total_felt_distance_traveled_base
    _, trdt_opt = total_distance_traveled_list(
        total_real_distance_traveled, total_real_distance_traveled_opt, buildup
    )
    tfdt_opt = total_felt_distance_traveled_opt
    trdt, trdt_cs = total_distance_traveled_list(
        total_real_distance_traveled, total_real_distance_traveled_cs, buildup
    )
    tfdt_cs = total_felt_distance_traveled_cs
    tfdt_ex_inf = total_felt_distance_traveled_ex_inf

    if buildup:
        bpp = bike_path_perc
        tfdt = total_felt_distance_traveled
    else:
        bpp = list(reversed(bike_path_perc))
        tfdt = list(reversed(total_felt_distance_traveled))

    trdt_min = trdt_opt["all"]
    trdt_max = trdt_base["all"]
    tfdt_min = tfdt_opt
    tfdt_max = tfdt_base

    ba = [(tfdt_max - i) / (tfdt_max - tfdt_min) for i in tfdt]
    ba_cs = (tfdt_max - tfdt_cs) / (tfdt_max - tfdt_min)
    ba_ex_inf = (tfdt_max - tfdt_ex_inf) / (tfdt_max - tfdt_min)
    max_ba = max(max(ba), ba_cs, ba_ex_inf)

    if buildup:
        nos = [x / max(nbr_on_street) for x in nbr_on_street]
    else:
        nos = list(reversed([x / max(nbr_on_street) for x in nbr_on_street]))
    nos_cs = nos_cs / max(nbr_on_street)
    los = trdt["street"]
    los_cs = trdt_cs["street"]

    trdt_st = {
        st: len_on_st for st, len_on_st in trdt.items() if st not in ["street", "all"]
    }
    trdt_st_cs = {
        st: len_on_st
        for st, len_on_st in trdt_cs.items()
        if st not in ["street", "all"]
    }

    total_cost = sum_total_cost(cost, buildup)

    if end_plot != len(bpp):
        bpp_normed = [i / bpp[end_plot] for i in bpp]
        bpp_cs = bike_path_perc_cs / bpp[end_plot]
        bpp_ex_inf = bike_path_perc_ex_inf / bpp[end_plot]
        max_bpp = max(bpp_normed[end_plot], bpp_cs)
        total_cost_normed = [x / total_cost[end_plot] for x in total_cost]
        cost_cs = cost_cs / total_cost[end_plot]
    else:
        bpp_normed = bpp
        bpp_cs = bike_path_perc_cs
        bpp_ex_inf = bike_path_perc_ex_inf
        max_bpp = bpp[-1]
        total_cost_normed = [x / total_cost[-1] for x in total_cost]
        cost_cs = cost_cs

    bpp_x = min(bpp_normed, key=lambda x: abs(x - bpp_cs))
    bpp_idx = next(x for x, val in enumerate(bpp_normed) if val == bpp_x)
    ba_y = ba[bpp_idx]

    bpp_20 = min(bpp_normed, key=lambda x: abs(x - 0.2))
    bpp_20_idx = next(x for x, val in enumerate(bpp_normed) if val == bpp_20)
    ba_20 = ba[bpp_20_idx]

    ba_cp_2 = min(ba, key=lambda x: abs(x - ba_cs))
    ba_cp_2_idx = next(x for x, val in enumerate(ba) if val == ba_cp_2)
    bpp_cp_2 = bpp_normed[ba_cp_2_idx]

    nos_y = nos[bpp_idx]
    los_y = los[bpp_idx]

    print(
        f"Mode: {buildup:d}{minmode}{ex_inf:d}, max ba after: {end_plot:d}, "
        f"bpp at max ba: {max_bpp:3.2f}, "
        f"bpp big roads: {bpp_cs * max_bpp:3.2f}, "
        f"edges: {len(edited_edges)}, max bpp: {max_bpp:3.2f}"
    )

    print(
        f"Reached 80% of max ba at "
        f"{bpp[next(x for x, val in enumerate(ba) if val >= 0.8*max_ba)]:3.2f} bpp"
    )
    print(
        f"Reached 80% of max ba at "
        f"{bpp_normed[next(x for x, val in enumerate(ba) if val >= 0.8*max_ba)]:3.2f}"
        f" normed bpp"
    )

    print(f"Minimal bpp: {min(bpp):3.2f}")
    print(f"Minimal normed bpp: {min(bpp_normed):3.2f}")

    # Plotting
    save_path_ba = join(
        plot_folder,
        f'{save}_ba_20_mode_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
    )
    plot_ba_cost(
        bpp_normed,
        ba,
        total_cost_normed,
        [(bpp_20, ba_20)],
        [],
        save_path_ba,
        plot_cost=False,
        ex_inf=ex_inf,
        x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
        params=params,
    )

    save_path_ba = join(
        plot_folder,
        f'{save}_ba_tc_mode_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
    )
    plot_ba_cost(
        bpp_normed,
        ba,
        total_cost_normed,
        [(bpp_cs, ba_cs), (bpp_cp_2, ba_cp_2)],
        [(bpp_cs, cost_cs)],
        save_path_ba,
        ex_inf=ex_inf,
        x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
        params=params,
    )

    save_path_ba = join(
        plot_folder,
        f'{save}_ba_tc_zoom_mode_{buildup:d}{minmode}{ex_inf:d}.'
        f'{params["plot_format"]}',
    )
    plot_ba_cost(
        bpp_normed,
        ba,
        total_cost_normed,
        [(bpp_cs, ba_cs), (bpp_cp_2, ba_cp_2)],
        [(bpp_cs, cost_cs)],
        save_path_ba,
        ba_diff_zoom=True,
        ex_inf=ex_inf,
        x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
        params=params,
    )

    save_path_ba = join(
        plot_folder,
        f'{save}_ba_tc_econconst_mode_{buildup:d}{minmode}{ex_inf:d}.'
        f'{params["plot_format"]}',
    )
    plot_ba_cost(
        bpp_normed,
        ba,
        total_cost_normed,
        [],
        [],
        save_path_ba,
        eco_opt_bpp=True,
        ex_inf=ex_inf,
        x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
        params=params,
    )

    save_path_los_nos = join(
        plot_folder,
        f'{save}_trips_on_street_mode_{buildup:d}{minmode}{ex_inf:d}.'
        f'{params["plot_format"]}',
    )
    plot_los_nos(
        bpp_normed,
        los,
        nos,
        [(bpp_cs, los_cs)],
        [(bpp_cs, nos_cs)],
        save_path_los_nos,
        x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
        params=params,
    )

    comp_st_driven = {
        st: [len_on_st[bpp_idx], trdt_st_cs[st]] for st, len_on_st in trdt_st.items()
    }
    plot_barv_stacked(
        ["Algorithm", "P+S"],
        comp_st_driven,
        params["c_st"],
        width=0.3,
        title="",
        save=join(
            plot_folder,
            f'{save}_comp_st_driven_{buildup:d}{minmode}{ex_inf:d}.'
            f'{params["plot_format"]}',
        ),
        figsize=params["figs_comp_st"],
    )

    if params["plot_bp_comp"] and ex_inf:
        bp_comp_modes = ["ex_inf", "algo", "cs", "diff"]
    elif params["plot_bp_comp"] and not ex_inf:
        bp_comp_modes = ["ex_inf", "algo", "cs", "diff", "diff_ex_inf"]
    else:
        bp_comp_modes = []
    for bp_mode in bp_comp_modes:
        plot_bp_comparison(
            city=city,
            save=save,
            G=G,
            ee_algo=edited_edges,
            ee_cs=bike_paths_cs,
            bpp_algo=bike_path_perc,
            bpp_cs=bike_path_perc_cs,
            buildup=buildup,
            minmode=minmode,
            ex_inf=ex_inf,
            plot_folder=plot_folder,
            mode=bp_mode,
            params=params,
            edge_loads_algo=edge_loads,
            edge_load_cs=data_cs["edge_loads"],
        )

    save_path_ba = join(
        plot_folder,
        f'{save}_ba_tc_prop_mode_{buildup:d}{minmode}{ex_inf:d}.'
        f'{params["plot_format"]}',
    )
    if ex_inf:
        plot_ba_cost(
            bpp_normed,
            ba,
            total_cost_normed,
            [(min(bpp_normed), min(ba))],
            [],
            save_path_ba,
            plot_cost=False,
            ex_inf=ex_inf,
            x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
            params=params,
        )
    else:
        plot_ba_cost(
            bpp_normed,
            ba,
            total_cost_normed,
            [(bpp_ex_inf, ba_ex_inf)],
            [],
            save_path_ba,
            plot_cost=False,
            ex_inf=ex_inf,
            x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
            params=params,
        )
    if not ex_inf and params["plot_bp_comp"]:
        plot_bp_comparison(
            city=city,
            save=save,
            G=G,
            ee_algo=edited_edges,
            ee_cs=None,
            bpp_algo=bike_path_perc,
            bpp_cs=bpp_ex_inf,
            buildup=buildup,
            minmode=minmode,
            ex_inf=ex_inf,
            save_name=f"{save}-bp-build-{buildup:d}{minmode}{ex_inf:d}_prop",
            plot_folder=plot_folder,
            mode="algo",
            params=params,
            edge_loads_algo=edge_loads,
            edge_load_cs=data_cs["edge_loads"],
        )
    elif ex_inf and params["plot_bp_comp"]:
        total_network_len = sum([d["length"] for u, v, d in G.edges(data=True)])
        bike_path_len_ex_inf = bike_path_perc_ex_inf * total_network_len

        bike_path_len_added = 20000
        bpp_prop = bike_path_len_added / total_network_len + bike_path_perc_ex_inf
        bike_path_len_prop = bpp_prop * total_network_len

        print(f"Bike Path length ex inf: {bike_path_len_ex_inf / 1000:3.2f} km")
        print(f"Bike Path length prop: {bike_path_len_prop / 1000:3.2f} km")
        print(f"Bike Path length added: {bike_path_len_added / 1000:3.2f} km")
        plot_bp_comparison(
            city=city,
            save=save,
            G=G,
            ee_algo=edited_edges,
            ee_cs=None,
            bpp_algo=bike_path_perc,
            bpp_cs=bpp_prop,
            buildup=buildup,
            minmode=minmode,
            ex_inf=ex_inf,
            save_name=f"{save}-bp-build-{buildup:d}{minmode}{ex_inf:d}_prop",
            plot_folder=plot_folder,
            mode="algo",
            params=params,
            edge_loads_algo=edge_loads,
            edge_load_cs=data_cs["edge_loads"],
        )

    if evo:
        plot_bp_evo(
            save=save,
            G=G,
            edited_edges=edited_edges,
            bpp=bpp_normed,
            buildup=buildup,
            minmode=minmode,
            ex_inf=ex_inf,
            plot_folder=plot_folder,
            params=params,
        )

    plt.close("all")

    print(
        f"ba: {ba_y:4.3f}, ba comp. state: {ba_cs:4.3f}, "
        f"los: {los_y:4.3f}, los comp. state: {los_cs:4.3f}, "
        f"nos: {nos_y:4.3f}, nos comp. state: {nos_cs:4.3f}, "
        f"bpp comp. state: {bpp_cs:4.3f}"
    )
    print(
        f"diff ba: {ba_y - ba_cs:4.3f}, "
        f"diff los: {los_cs - los_y:4.3f}, "
        f"diff nos: {nos_cs - nos_y:4.3f}, "
    )

    evaluation_data["edited edges"] = edited_edges
    evaluation_data["end"] = end_plot
    evaluation_data["bpp normed"] = bpp_normed
    evaluation_data["bpp at end"] = max_bpp
    evaluation_data["bpp complete"] = bpp
    evaluation_data["ba"] = ba
    evaluation_data["ba for comp"] = ba_y
    evaluation_data["ba for ex_inf"] = ba_ex_inf
    evaluation_data["cost normed"] = total_cost_normed
    evaluation_data["cost complete"] = total_cost
    evaluation_data["number_on_street"] = nos
    evaluation_data["nos at comp"] = nos_y
    evaluation_data["los"] = los
    evaluation_data["los at comp"] = los_y
    evaluation_data["total_felt_distance_traveled"] = tfdt
    evaluation_data["total_felt_distance_traveled_max"] = tfdt_max
    evaluation_data["total_felt_distance_traveled_min"] = tfdt_min
    evaluation_data["total_real_distances_traveled"] = trdt["all"]
    evaluation_data["total_real_distance_traveled_max"] = trdt_max
    evaluation_data["total_real_distance_traveled_min"] = trdt_min

    return bpp_cs, ba_cs, cost_cs, nos_cs, los_cs


def plot_city(
    city: str,
    save: str,
    modes: list[tuple[bool, int, bool]] | None = None,
    params: dict | None = None,
    paths: dict | None = None,
):
    """Plots the results for the given city. If no 'paths' or 'params' are given
    the default ones are created and used.

    Parameters
    ----------
    city : str
        Name of the city/area
    save : str
        Save name of the city/area
    modes :
        type modes: (Default value = None)
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict
        Paths for data, plots etc., check 'setup_paths.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if modes is None:
        modes = [(False, 1, False)]
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()

    # Define city specific folders
    comp_folder = paths["comp_folder"]
    plot_folder = join(paths["plot_folder"], "results", save)
    input_folder = join(paths["input_folder"], save)
    output_folder = join(paths["output_folder"], save)

    # Create non existing folders
    Path(comp_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    data_comp = dict()
    data_comp["city"] = city

    plt.rcdefaults()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    if paths["demand_file"] is None:
        paths["demand_file"] = join(input_folder, f"{save}_demand.json")
    demand = load_demand(paths["demand_file"])
    trip_nbrs = {literal_eval(k): v for k, v in demand.items()}
    trip_nbrs_re = {
        trip_id: nbr_of_trips
        for trip_id, nbr_of_trips in trip_nbrs.items()
        if not trip_id[0] == trip_id[1]
    }
    trips = sum([t for trip in trip_nbrs.values() for t in trip.values()])
    trips_re = sum([t for trip in trip_nbrs_re.values() for t in trip.values()])
    data_comp["total trips"] = trips_re
    data_comp["total trips (incl round trips)"] = trips
    utrips = len(trip_nbrs.keys())
    utrips_re = len(trip_nbrs_re.keys())
    data_comp["unique trips"] = utrips_re
    data_comp["unique trips (incl round trips)"] = utrips

    stations = [
        station for trip_id, nbr_of_trips in trip_nbrs.items() for station in trip_id
    ]
    stations = list(set(stations))
    data_comp["nbr of stations"] = len(stations)
    data_comp["stations"] = stations

    if paths["graph_file"] is None:
        paths["graph_file"] = join(input_folder, f"{save}.graphml")
    G = ox.load_graphml(
        filepath=paths["graph_file"],
        node_dtypes={"street_count": float},
    )
    data_comp["nodes"] = len(G.nodes)
    data_comp["edges"] = len(G.edges)

    print(
        f"City: {city}, "
        f"nodes: {len(G.nodes)}, edges: {len(G.edges)}, "
        f"stations: {len(stations)}, "
        f"unique trips: {utrips_re} (rt incl.: {utrips}), "
        f"trips: {trips_re} (rt incl.: {trips})"
    )
    st_ratio = get_street_type_ratio(G)
    data_comp["ratio street type"] = st_ratio
    sn_ratio = len(stations) / len(G.nodes())
    data_comp["ratio stations nodes"] = sn_ratio

    if paths["polygon_file"] is None:
        paths["polygon_file"] = join(paths["polygon_folder"], f"{save}.json")
    polygon = get_polygon_from_json(paths["polygon_file"])

    area = calc_polygon_area(polygon)
    data_comp["area"] = area
    print(f"Area {round(area, 1)}")
    sa_ratio = len(stations) / area
    data_comp["ratio stations area"] = sa_ratio

    plot_used_nodes(
        city=city,
        save=save,
        G=G,
        trip_nbrs=trip_nbrs,
        stations=stations,
        plot_folder=plot_folder,
        params=params,
    )

    data_algo = {}
    cs_bike_paths = []
    for m in modes:
        data_algo[m] = load_algorithm_results(
            join(output_folder, f"{save}_data_mode_{m[0]:d}{m[1]}{m[2]:d}.json")
        )
        if len(cs_bike_paths) < len(data_algo[m]["used_primary_secondary_edges"]):
            cs_bike_paths = data_algo[m]["used_primary_secondary_edges"]

    data_base = load_comparison_state_results(
        join(output_folder, f"{save}_data_comparison_state_base.json")
    )
    data_opt = load_comparison_state_results(
        join(output_folder, f"{save}_data_comparison_state_opt.json")
    )
    data_ex_inf = load_comparison_state_results(
        join(output_folder, f"{save}_data_comparison_state_ex_inf.json")
    )
    data_cs = load_comparison_state_results(
        join(output_folder, f"{save}_data_comparison_state_cs.json")
    )
    data_cs_ex_inf = load_comparison_state_results(
        join(output_folder, f"{save}_data_comparison_state_cs_ex_inf.json")
    )

    data_comp["algorithm"] = dict()
    grp_algo = data_comp["algorithm"]
    for m, d in data_algo.items():
        mode_str = f"{m[0]:d}{m[1]}{m[2]:d}"
        if params["cut"]:
            end_plot = d["cut"]
        else:
            end_plot = len(d["bike_path_percentage"])
        print(f"Mode {mode_str}: cut after {end_plot}")

        grp_algo[mode_str] = dict()
        sbgrp_algo = grp_algo[mode_str]
        if m[2]:
            print(f"CS use ex_inf")
            data_cs_use = data_cs_ex_inf
        else:
            data_cs_use = data_cs
        if params["plot_evo"] and (m in params["evo_for"]):
            evo = True
        else:
            evo = False
        bpp_cs, ba_cs, cost_cs, nos_cs, los_cs = plot_mode(
            city=city,
            save=save,
            data=d,
            data_cs=data_cs_use,
            data_base=data_base,
            data_opt=data_opt,
            data_ex_inf=data_ex_inf,
            G=G,
            mode=m,
            end_plot=end_plot,
            evo=evo,
            evaluation_data=sbgrp_algo,
            plot_folder=plot_folder,
            params=params,
            paths=paths,
        )
        if m[2]:
            if "cs ex_inf" not in data_comp.keys():
                data_comp["cs ex_inf"] = dict()
                grp_ps = data_comp["cs ex_inf"]
                grp_ps["bike_path_percentage"] = bpp_cs
                grp_ps["ba"] = ba_cs
                grp_ps["cost"] = cost_cs
                grp_ps["number_on_street"] = nos_cs
                grp_ps["los"] = los_cs
        else:
            if "cs" not in data_comp.keys():
                data_comp["cs"] = dict()
                grp_ps = data_comp["cs"]
                grp_ps["bike_path_percentage"] = bpp_cs
                grp_ps["ba"] = ba_cs
                grp_ps["cost"] = cost_cs
                grp_ps["number_on_street"] = nos_cs
                grp_ps["los"] = los_cs

    save_algorithm_results(data_comp, join(comp_folder, f"comp_{save}.json"))


def comp_city(
    city: str,
    base_save: str,
    factors: list[str],
    mode: tuple[bool, int, bool],
    labels: dict | None = None,
    params: dict | None = None,
    paths: dict | None = None,
):
    """Compare bikeability for same city with different factors. All saved data should
    have the same structure 'base_save'+'factor'.

    Parameters
    ----------
    city : str
        Name of the city/area
    base_save : str
        Base save structure of the city/area
    factors : list[str]
        Save names of the influencing structure
    mode : tuple[bool, int, bool]
        Mode for comparison
    labels : dict | None
        Labels for the different factors
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict
        Paths for data, plots etc., check 'setup_paths.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()
    if labels is None:
        labels = {factor: factor for factor in factors}
    comp_folder = paths["comp_folder"]
    mode_save = f"{mode[0]:d}{mode[1]}{mode[2]:d}"

    bpps = dict()
    bpps_end = dict()
    bas = dict()
    ends = dict()

    for factor in factors:
        if factor != "":
            save = f"{base_save}_{factor}"
        else:
            save = base_save
        data = load_algorithm_results(join(comp_folder, f"comp_{save}.json"))
        data_grp = data["algorithm"][mode_save]
        bpps[factor] = data_grp["bpp complete"]
        bpps_end[factor] = data_grp["bpp at end"]
        bas[factor] = data_grp["ba"]
        ends[factor] = data_grp["end"]

    end = max(ends.values())

    for factor in factors:
        bpps[factor] = [x / bpps[factor][end] for x in bpps[factor][: end + 1]]

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_ylabel(r"bikeability b($\lambda$)", fontsize=params["fs_axl"])
    ax.set_xlabel(
        r"normalized relative length of bike paths $\lambda$",
        fontsize=params["fs_axl"],
    )

    ax.tick_params(axis="y", labelsize=params["fs_ticks"])
    ax.tick_params(axis="x", labelsize=params["fs_ticks"])

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    cmap = colormaps["viridis"]

    colors = [cmap(i) for i in np.linspace(0, 1, num=len(factors))]
    colors[0] = "k"
    # colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for idx, factor in enumerate(factors):
        ax.plot(
            bpps[factor],
            bas[factor][: end + 1],
            label=labels[factor],
            c=colors[idx],
            lw=params["lw_ba"],
        )

    if params["titles"]:
        ax.set_title(
            f"Bikeability for {city.capitalize()}", fontsize=params["fs_title"]
        )
    if params["legends"]:
        ax.legend(loc="lower right", fontsize=params["fs_legend"], frameon=False)

    fig.savefig(
        join(
            paths["plot_folder"],
            "results",
            base_save,
            f'{base_save}_ba_factor_comp_{mode_save}.{params["plot_format"]}',
        ),
        bbox_inches="tight",
    )


def comp_modes(
    city: str,
    save: str,
    modes: list[tuple[bool, int, bool]],
    params: dict | None = None,
    paths: dict | None = None,
):
    """Compare the bikeability different algorithm modes for the same data set.

    Parameters
    ----------
    city : str
        Name of the city/area
    save : str
        Save name of the city/area
    modes : list[tuple[bool, int, bool]]
        Modes to compare.
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict
        Paths for data, plots etc., check 'setup_paths.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------

    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()
    comp_folder = paths["comp_folder"]

    bpps = dict()
    bas = dict()
    costs = dict()
    ends = dict()

    for mode in modes:
        mode_save = f"{mode[0]:d}{mode[1]}{mode[2]:d}"
        data = load_algorithm_results(join(comp_folder, f"comp_{save}.json"))
        data_grp = data["algorithm"][mode_save]
        bpps[mode_save] = data_grp["bpp complete"]
        bas[mode_save] = data_grp["ba"]
        costs[mode_save] = data_grp["cost complete"]
        ends[mode_save] = data_grp["end"]

    end = max(ends.values())
    max_cost = max([costs[f"{mode[0]:d}{mode[1]}{mode[2]:d}"][end] for mode in modes])

    for mode in modes:
        mode_save = f"{mode[0]:d}{mode[1]}{mode[2]:d}"
        bpp = [x / bpps[mode_save][end] for x in bpps[mode_save][: end + 1]]
        bpps[mode_save] = bpp
        costs[mode_save] = [i / max_cost for i in costs[mode_save]]

    bpps_comp = dict()
    bpps_80 = dict()
    bas_80 = dict()
    for mode in modes:
        mode_save = f"{mode[0]:d}{mode[1]}{mode[2]:d}"
        if mode_save != "010":
            ba_y = min(bas["010"], key=lambda y: abs(y - bas[mode_save][0]))
            ba_idx = next(x for x, val in enumerate(bas["010"]) if val == ba_y)
            bpps_comp[mode_save] = bpps["010"][ba_idx]
        ba_80 = min(bas[mode_save], key=lambda y: abs(y - 0.8))
        ba_80_idx = next(x for x, val in enumerate(bas[mode_save]) if val == ba_80)
        bas_80[mode_save] = ba_80
        bpps_80[mode_save] = bpps[mode_save][ba_80_idx]

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_ylabel(r"bikeability b($\lambda$)", fontsize=params["fs_axl"])
    ax.set_xlabel(
        r"normalized relative length of bike paths $\lambda$",
        fontsize=params["fs_axl"],
    )

    ax.tick_params(axis="y", labelsize=params["fs_ticks"])
    ax.tick_params(axis="x", labelsize=params["fs_ticks"])

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    colors = ["C0", "C1", "C2", "C3"]

    for idx, mode in enumerate(modes):
        mode_save = f"{mode[0]:d}{mode[1]}{mode[2]:d}"
        ax.plot(
            bpps[mode_save],
            bas[mode_save][: end + 1],
            label=mode_save,
            c=colors[idx],
            lw=params["lw_ba"],
        )

        ax.axvline(
            x=bpps_80[mode_save],
            ymax=0.8,
            ymin=0,
            c="#808080",
            ls="--",
            alpha=0.5,
            lw=params["lw_ba"],
        )
        print(
            f"{save} {mode_save}: Reached ba=0.8 at "
            f"{bpps_80[mode_save]:3.2f}"
            f" normed bpp"
        )

    ax.axhline(
        y=0.8,
        xmax=max(bpps_80.values()),
        xmin=0,
        c="#808080",
        ls="--",
        alpha=0.5,
        lw=params["lw_ba"],
    )

    if params["titles"]:
        ax.set_title(
            f"Bikeability for {city.capitalize()}", fontsize=params["fs_title"]
        )
    if params["legends"]:
        ax.legend(loc="lower right", fontsize=params["fs_legend"], frameon=False)

    fig.savefig(
        join(
            paths["plot_folder"],
            "results",
            save,
            f'{save}_ba_mode_comp.{params["plot_format"]}',
        ),
        bbox_inches="tight",
    )


def plot_city_comparison(
    cities: dict,
    figsave: str = "comp_cities",
    mode: tuple[bool, int, bool] = (False, 1, False),
    params: dict | None = None,
    paths: dict | None = None,
):
    """Compare multiple cities for one mode.

    Parameters
    ----------
    cities : dict
        Cities to compare {name: save}.
    figsave : str
        Name of the figure file. (Default value = "comp_cities")
    mode :
         Mode for comparison. (Default value = (False, 1, False))
    params : dict | None
        Params for plots etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict | None
        Dict with the paths for data, plots etc., check 'setup_paths.py' in the
        'scripts' folder. (Default value = None)
    Returns
    -------

    """
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    plt.rcdefaults()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    comp_folder = paths["comp_folder"]

    bpp = {}
    ba = {}

    for city, save in cities.items():
        data_city = load_algorithm_results(join(comp_folder, f"comp_{save}.json"))
        bpp[city] = data_city["algorithm"][mode]["bpp normed"]
        ba[city] = data_city["algorithm"][mode]["ba"]

    fig1, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
    max_y = max([max(ba_c) for ba_c in ba.values()])
    min_y = min([min(ba_c) for ba_c in ba.values()])
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(min([0.0, min_y]), max([1.0, max_y]))

    if isinstance(params["cmap_city_comp"], ListedColormap):
        cmap = params["cmap_city_comp"]
    elif isinstance(params["cmap_city_comp"], str):
        cmap = colormaps[params["cmap_city_comp"]]
    else:
        print(
            "Colormap must be colormap obbject or string. "
            "Defaulting to viridis colormap."
        )
        cmap = colormaps["viridis"]

    for idx, (city, save) in enumerate(cities.items()):
        ax1.plot(
            bpp[city],
            ba[city],
            color=cmap(idx / len(cities)),
            label=f"{city}",
            lw=params["lw_ba"],
        )

    ax1.set_ylabel(r"bikeability b($\lambda$)", fontsize=params["fs_axl"])
    ax1.set_xlabel(
        r"normalized relative length of bike paths $\lambda$",
        fontsize=params["fs_axl"],
    )
    ax1.tick_params(axis="y", labelsize=params["fs_ticks"], width=0.5)
    ax1.tick_params(axis="x", labelsize=params["fs_ticks"], width=0.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    if params["legends"]:
        ax1.legend(
            bbox_to_anchor=(1.0, 1.0),
            fontsize=params["fs_legend"],
            frameon=False,
        )

    fig1.savefig(
        join(
            paths["plot_folder"], "results", f'{figsave}_{mode}.{params["plot_format"]}'
        ),
        bbox_inches="tight",
    )
    plt.close()
