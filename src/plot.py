"""
This module includes all necessary functions for the plotting functionality.
"""
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from ast import literal_eval
from os.path import join
from scipy import interpolate, optimize
from matplotlib.colors import rgb2hex, ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
from pathlib import Path
from .data import (
    get_polygon_from_json,
    get_polygons_from_json,
)
from .algorithm import calc_comparison_state
from .plot_helper import *
from .setup_helper import create_default_paths, create_default_params


def plot_ba_cost(
    bpp,
    ba,
    cost,
    ba_comparisons,
    cost_comparisons,
    save_path,
    plot_cost=True,
    ba_diff_zoom=False,
    eco_opt_bpp=False,
    ex_inf=False,
    x_min=0.0,
    x_max=1.0,
    y_min=0.0,
    y_max=1.0,
    params=None,
):
    """ """
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
        "bikeability b($\lambda$)",
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

        threshold = 0
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

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_los_nos(
    bpp,
    los,
    nos,
    los_comparisons,
    nos_comparisons,
    save_path,
    x_min=0.0,
    x_max=1.0,
    y_min=0.0,
    y_max=1.0,
    params=None,
):
    """ """
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


def get_comparison_edge_load(
    save, bpp_algo, bpp_cs, bike_paths, exinf, buildup, params, paths, minmode=1
):
    """ """
    if buildup:
        bike_paths = bike_paths
        bpp_algo = bpp_algo
    else:
        bike_paths = list(reversed(bike_paths))
        bpp_algo = list(reversed(bpp_algo))

    idx = min(range(len(bpp_algo)), key=lambda i: abs(bpp_algo[i] - bpp_cs))

    data = calc_comparison_state(
        save,
        f"algo_comp_edge_load_{buildup:d}{minmode}{exinf:d}",
        bike_paths=bike_paths[:idx],
        ex_inf=exinf,
        use_penalties=True,
        base_state=False,
        opt_state=False,
        params=params,
        paths=paths,
    )

    return data["edge_load"]


def plot_street_network(city, save, G, plot_folder, params=None):
    """
    Plots the street network of graph G.
    :param city: Name of the city/area
    :type city: str
    :param save: Save name of the city/area
    :type save: str
    :param G: Graph of the ares
    :type G: osmnx graph
    :param plot_folder: Folder to save the plot
    :type plot_folder: str
    :param params: Dict with parameters
    :type params: dict or None
    """
    fig, ax = plt.subplots(figsize=params["figs_snetwork"], dpi=params["dpi"])
    ox.plot_graph(
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


def plot_used_nodes(city, save, G, trip_nbrs, stations, plot_folder, params=None):
    """
    Plots usage of nodes in graph G. trip_nbrs and stations should be
    structured as returned from load_trips().
    :param city: Name of the city/area
    :type city: str
    :param save: Save name of the city/area
    :type save: str
    :param G: Graph of the ares
    :type G: osmnx graph
    :param trip_nbrs: Dict with trips and number of cyclists
    :type trip_nbrs: dict
    :param stations: List of nodes used as stations
    :type stations: list
    :param plot_folder: Folder to save the plot
    :type plot_folder: str
    :param params: Dict with parameters
    :type params: dict or None
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

    max_load = max(node_load.values())
    print(f"Maximal station usage: {max_load}")
    min_load = min([load for n, load in node_load.items() if n in stations])
    print(f"Minimal station usage: {min_load}")

    r = magnitude(max_load)

    hist_data = [load for n, load in node_load.items() if n in stations]
    hist_save = join(
        plot_folder, f'{save}_stations_usage_distribution.{params["plot_format"]}'
    )
    hist_xlim = (0.0, round(max_load, -(r - 1)))

    cmap = plt.cm.get_cmap(params["cmap_nodes"])

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
    node_size = [params["nodesize"] if n in stations else 0 for n in G.nodes()]
    node_load_normed = {
        n: (load - vmin) / max_load if n in stations else 0
        for n, load in node_load.items()
    }
    node_color = [rgb2hex(cmap(node_load_normed[n])) for n in G.nodes()]

    fig2, ax2 = plt.subplots(dpi=params["dpi"], figsize=params["figs_station_usage"])
    ox.plot_graph(
        G,
        ax=ax2,
        bgcolor="#ffffff",
        node_size=node_size,
        node_color=node_color,
        edge_linewidth=params["ew_snetwork"],
        edge_color=params["ec_station_usage"],
        node_zorder=3,
        show=False,
        close=False,
    )

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap(params["cmap_nodes"]),
        norm=plt.Normalize(vmin=vmin, vmax=max_load),
    )

    cbaxes = fig2.add_axes([0.1, 0.05, 0.8, 0.03])
    # divider = make_axes_locatable(ax2)
    # cbaxes = divider.append_axes('bottom', size="5%", pad=0.05)

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
        orientation="horizontal",
        cax=cbaxes,
        ticks=[vmin, (max_load - vmin) / 2, max_load],
    )
    cbar.ax.set_xticklabels([f_tick, m_tick, l_tick])

    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(axis="x", labelsize=params["fs_ticks"], width=0.5)
    cbar.ax.set_xlabel("total number of trips per year", fontsize=params["fs_axl"])

    if params["titles"]:
        fig2.suptitle(f"{city.capitalize()}", fontsize=params["fs_title"])
        """ax2.set_title(f"Stations: {station_count}, Trips: {trip_count}",
                      fontsize=params["fs_axl"])"""

    fig2.savefig(
        join(plot_folder, f'{save}_stations_used.{params["plot_format"]}'),
        bbox_inches="tight",
    )

    plt.close("all")


def plot_edge_load(
    G,
    edge_load,
    save,
    buildup,
    minmode,
    ex_inf,
    mode,
    plot_folder,
    params,
):
    """
    Plotting the edge load of the network for a given step.
    :param G: Graph to plot the edge load of
    :type G: Networkx (Multi)(Di)Graph
    :param edge_load: Dict with edg load during the algorithm. key: edge,
                        value: list with load per step
    :type edge_load: dict
    :param save:
    :param buildup:
    :param minmode:
    :param ex_inf:
    :param mode:
    :param plot_folder: Folder to save the plot
    :type plot_folder: str
    :param params: Dict with parameters
    :type params: dict or None
    :return:
    """
    nx.set_edge_attributes(G, 0, "load")

    edge_load = {eval(k): v for k, v in edge_load.items()}
    edge_load = {(k[0], k[1], 0): v for k, v in edge_load.items()}
    nx.set_edge_attributes(G, edge_load, "load")

    fig, ax = plt.subplots(dpi=600, figsize=params["figs_bp_comp"])
    ax.set_facecolor("k")
    ec = ox.plot.get_edge_colors_by_attr(G, "load", cmap="magma")

    ox.plot_graph(
        G,
        bgcolor="#ffffff",
        ax=ax,
        node_size=0,
        edge_color=ec,
        edge_linewidth=params["ew_snetwork"],
        show=False,
        close=False,
    )

    plt.savefig(
        join(
            plot_folder,
            f'{save}-load-{buildup:d}{minmode}{ex_inf:d}_{mode}.{params["plot_format"]}',
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_bp_comparison(
    city,
    save,
    G,
    ee_algo,
    ee_cs,
    bpp_algo,
    bpp_cs,
    stations,
    buildup,
    minmode,
    ex_inf,
    plot_folder,
    save_name=None,
    mode="diff",
    params=None,
    edge_loads_algo=None,
    edge_load_cs=None,
    idx=None,
):
    """ """
    if edge_load_cs is None:
        edge_load_cs = dict()
    if edge_loads_algo is None:
        edge_loads_algo = dict()
    nx.set_edge_attributes(G, False, "algo")
    nx.set_edge_attributes(G, False, "cs")

    ns = [params["nodesize"] if n in stations else 0 for n in G.nodes()]

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
        params,
    )
    plot_edge_load(
        G, edge_load_cs, save, buildup, minmode, ex_inf, "cs", plot_folder, params
    )

    print(
        f"Difference in BPP between cs and algo: " f"{abs(bpp_cs - bpp_algo[idx]):4.3f}"
    )

    ee_algo_cut = ee_algo[:idx]
    for edge in ee_algo_cut:
        G[edge[0]][edge[1]][0]["algo"] = True
    for edge in ee_cs:
        G[edge[0]][edge[1]][0]["cs"] = True

    ec = []

    len_algo = 0
    len_cs = 0
    len_both = 0

    for u, v, data in G.edges(data=True):
        if not isinstance(data["ex_inf"], bool):
            data["ex_inf"] = literal_eval(data["ex_inf"])

    if mode == "algo":
        for u, v, data in G.edges(data=True):
            if data["algo"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                else:
                    ec.append(params["color_algo"])
            else:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                else:
                    ec.append(params["color_unused"])
    elif mode == "cs":
        for u, v, data in G.edges(data=True):
            if data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                else:
                    ec.append(params["color_cs"])
            else:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                else:
                    ec.append(params["color_unused"])
    elif mode == "diff":
        for u, v, data in G.edges(data=True):
            if data["algo"] and data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_both_ex_inf"])
                else:
                    ec.append(params["color_both"])
                len_both += data["length"]
            elif data["algo"] and not data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                else:
                    ec.append(params["color_algo"])
                len_algo += data["length"]
            elif not data["algo"] and data["cs"]:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                else:
                    ec.append(params["color_cs"])
                len_cs += data["length"]
            else:
                if ex_inf and data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                else:
                    ec.append(params["color_unused"])
        print(
            f"Overlap between p+s and algo: "
            f"{len_both / (len_cs + len_both + len_algo):3.2f}"
        )
    elif mode == "diff_ex_inf":
        for u, v, data in G.edges(data=True):
            if data["algo"] and data["cs"]:
                if data["ex_inf"]:
                    ec.append(params["color_both_ex_inf"])
                else:
                    ec.append(params["color_both"])
                len_both += data["length"]
            elif data["algo"] and not data["cs"]:
                if data["ex_inf"]:
                    ec.append(params["color_algo_ex_inf"])
                else:
                    ec.append(params["color_algo"])
                len_algo += data["length"]
            elif not data["algo"] and data["cs"]:
                if data["ex_inf"]:
                    ec.append(params["color_cs_ex_inf"])
                else:
                    ec.append(params["color_cs"])
                len_cs += data["length"]
            else:
                if data["ex_inf"]:
                    ec.append(params["color_both_ex_inf"])
                else:
                    ec.append(params["color_unused"])
        print(
            f"Overlap between p+s and algo: "
            f"{len_both / (len_cs + len_both + len_algo):3.2f}"
        )
    elif mode == "ex_inf":
        for u, v, data in G.edges(data=True):
            if data["ex_inf"]:
                ec.append(params["color_ex_inf"])
            else:
                ec.append(params["color_unused"])
    else:
        print("You have to choose between algo, p+s and diff.")

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_bp_comp"])
    ox.plot_graph(
        G,
        bgcolor="#ffffff",
        ax=ax,
        node_size=0,
        node_color=params["nc_bp_comp"],
        node_zorder=3,
        edge_linewidth=params["ew_snetwork"],
        edge_color=ec,
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
            ax.set_title(f"{city}: Algorithm", fontsize=params["fs_title"])
        elif mode == "cs":
            ax.set_title(f"{city}: Primary/Secondary", fontsize=params["fs_title"])
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
    G,
    edge_color,
    node_size,
    save_path,
    title="",
    figsize=(12, 12),
    params=None,
):
    """ """
    fig, ax = plt.subplots(dpi=params["dpi"], figsize=figsize)
    ox.plot_graph(
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
    save,
    G,
    edited_edges,
    bpp,
    buildup,
    minmode,
    ex_inf,
    plot_folder,
    params=None,
):
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
    save,
    G,
    edge_loads,
    bpp,
    node_size,
    buildup,
    minmode,
    ex_inf,
    plot_folder,
    params=None,
):
    """ """
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
        edge_load_evo = {eval(k): v for k, v in edge_load_evo.items()}
        edge_load_evo = {
            (k[0], k[1], 0): v / max_load for k, v in edge_load_evo.items()
        }
        nx.set_edge_attributes(G, edge_load_evo, "load")
        ec_evo = ox.plot.get_edge_colors_by_attr(G, "load", cmap="magma_r")

        save_path = join(
            plot_folder_evo,
            f'{save}-load-mode-{buildup:d}{minmode}{ex_inf:d}-{i}.{params["plot_format"]}',
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
    city,
    save,
    data,
    data_cs,
    data_base,
    data_opt,
    data_ex_inf,
    G,
    stations,
    mode,
    end_plot,
    hf_group,
    plot_folder,
    evo=False,
    params=None,
    paths=None,
):
    """
    Plots the results for one mode for the given city. If no 'params' are given
    the default ones are created and used.

    :param city: Name of the city
    :type city: str
    :param save: Save name of the city.
    :type save: str
    :param data:
    :type data:
    :param data_cs:
    :type data_cs:
    :param data_base:
    :type data_base:
    :param data_opt:
    :type data_opt:
    :param data_ex_inf:
    :type data_ex_inf:
    :param G:
    :type G: networkx (Multi)(Di)graph
    :param stations:
    :type stations
    :param trip_nbrs:
    :type trip_nbrs:
    :param mode:
    :type mode: tuple
    :param end_plot:
    :type end_plot:
    :param hf_group:
    :type hf_group:
    :param plot_folder:
    :type plot_folder: str
    :param evo:
    :type evo: bool
    :param params: Dict with the params for plots etc., check 'params.py' in
    the example folder.
    :type params: dict
    :param paths: Dict with the paths for plots etc.
    :type paths: dict
    :return: None
    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()

    buildup = mode[0]
    minmode = mode[1]
    ex_inf = mode[2]

    bike_paths_cs = data_cs["bike_paths"]
    bike_path_perc_cs = data_cs["bpp"]
    cost_cs = data_cs["cost"]
    total_real_distance_traveled_cs = data_cs["trdt"]
    total_felt_distance_traveled_cs = data_cs["tfdt"]
    nos_cs = data_cs["nos"]

    bike_path_perc_ex_inf = data_ex_inf["bpp"]
    cost_ex_inf = data_ex_inf["cost"]
    total_felt_distance_traveled_ex_inf = data_ex_inf["tfdt"]

    total_real_distance_traveled_base = data_base["trdt"]
    total_felt_distance_traveled_base = data_base["tfdt"]

    total_real_distance_traveled_opt = data_opt["trdt"]
    total_felt_distance_traveled_opt = data_opt["tfdt"]

    edited_edges = json.loads(data["ee"][()])
    bike_path_perc = data["bpp"][()]
    cost = data["cost"][()]
    total_real_distance_traveled = json.loads(data["trdt"][()])
    total_felt_distance_traveled = data["tfdt"][()]
    nbr_on_street = data["nos"][()]
    edge_loads = json.loads(data["edge_load"][()])
    if not isinstance(edge_loads, dict):
        edge_loads = get_comparison_edge_load(
            save,
            bike_path_perc,
            bike_path_perc_cs,
            edited_edges,
            ex_inf,
            buildup,
            params,
            paths,
            minmode=minmode,
        )

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
        max_bpp = max(bpp_normed[end_plot], bpp_cs)
        total_cost_normed = [x / total_cost[end_plot] for x in total_cost]
        cost_cs = cost_cs / total_cost[end_plot]
        cost_ex_inf = cost_ex_inf / total_cost[end_plot]
    else:
        bpp_normed = bpp
        bpp_cs = bike_path_perc_cs
        max_bpp = bpp[-1]
        total_cost_normed = [x / total_cost[-1] for x in total_cost]
        cost_cs = cost_cs

    bpp_x = min(bpp_normed, key=lambda x: abs(x - bpp_cs))
    bpp_idx = next(x for x, val in enumerate(bpp_normed) if val == bpp_x)
    ba_y = ba[bpp_idx]

    ba_cp_2 = min(ba, key=lambda x: abs(x - ba_cs))
    ba_cp_2_idx = next(x for x, val in enumerate(ba) if val == ba_cp_2)
    bpp_cp_2 = bpp_normed[ba_cp_2_idx]

    cost_y = total_cost_normed[bpp_idx]

    nos_y = nos[bpp_idx]

    los_y = los[bpp_idx]

    ns = [params["nodesize"] if n in stations else 0 for n in G.nodes()]

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
        f'{save}_ba_tc_zoom_mode_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
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
        f'{save}_ba_tc_econconst_mode_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
    )
    plot_ba_cost(
        bpp_normed,
        ba,
        total_cost_normed,
        [(bpp_cs, ba_cs), (bpp_cp_2, ba_cp_2)],
        [(bpp_cs, cost_cs)],
        save_path_ba,
        eco_opt_bpp=True,
        ex_inf=ex_inf,
        x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
        params=params,
    )

    save_path_los_nos = join(
        plot_folder,
        f'{save}_trips_on_street_mode_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
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
            f'{save}_comp_st_driven_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
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
            stations=stations,
            buildup=buildup,
            minmode=minmode,
            ex_inf=ex_inf,
            plot_folder=plot_folder,
            mode=bp_mode,
            params=params,
            edge_loads_algo=edge_loads,
            edge_load_cs=data_cs["edge_load"],
        )
    ba_ex_inf_y = min(ba, key=lambda x: abs(x - 0.8446205776427792))
    ba_ex_inf_idx = next(x for x, val in enumerate(ba) if val == ba_ex_inf_y)
    bpp_ex_inf = bpp_normed[ba_ex_inf_idx]

    bpp_ex_inf_x1 = min(
        bpp_normed, key=lambda x: abs(x - bike_path_perc_ex_inf / bpp[end_plot])
    )
    bpp_ex_inf_idx = next(x for x, val in enumerate(bpp_normed) if val == bpp_ex_inf_x1)
    save_path_ba = join(
        plot_folder,
        f'{save}_ba_tc_prop_mode_{buildup:d}{minmode}{ex_inf:d}.{params["plot_format"]}',
    )
    if ex_inf:
        use_idx = ba_ex_inf_idx
        plot_ba_cost(
            bpp_normed,
            ba,
            total_cost_normed,
            [(bpp_ex_inf, ba_ex_inf_y), (min(bpp_normed), min(ba))],
            [(bpp_ex_inf, total_cost_normed[ba_ex_inf_idx])],
            save_path_ba,
            ex_inf=ex_inf,
            x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
            params=params,
        )
    else:
        use_idx = bpp_ex_inf_idx
        plot_ba_cost(
            bpp_normed,
            ba,
            total_cost_normed,
            [(bpp_ex_inf_x1, ba[bpp_ex_inf_idx])],
            [(bpp_ex_inf_x1, cost_ex_inf)],
            save_path_ba,
            ex_inf=ex_inf,
            x_max=max(1.0, math.ceil(max_bpp * 10) / 10),
            params=params,
        )
    if params["plot_bp_comp"]:
        plot_bp_comparison(
            city=city,
            save=save,
            G=G,
            ee_algo=edited_edges,
            ee_cs=bike_paths_cs,
            bpp_algo=bike_path_perc,
            bpp_cs=bike_path_perc_cs,
            stations=stations,
            buildup=buildup,
            minmode=minmode,
            ex_inf=ex_inf,
            save_name=f"{save}-bp-build-{buildup:d}{minmode}{ex_inf:d}_prop",
            plot_folder=plot_folder,
            mode="algo",
            params=params,
            edge_loads_algo=edge_loads,
            edge_load_cs=data_cs["edge_load"],
            idx=use_idx,
        )

    if evo:
        plot_bp_evo(
            save=save,
            G=G,
            edited_edges=edited_edges,
            bpp=bpp_normed,
            node_size=ns,
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

    hf_group["edited edges"] = edited_edges
    hf_group["end"] = end_plot
    hf_group["bpp normed"] = bpp_normed
    hf_group["bpp at end"] = max_bpp
    hf_group["bpp complete"] = bpp
    hf_group["ba"] = ba
    hf_group["ba for comp"] = ba_y
    hf_group["ba for ex_inf"] = ba_ex_inf
    hf_group["cost normed"] = total_cost_normed
    hf_group["cost complete"] = total_cost
    hf_group["nos"] = nos
    hf_group["nos at comp"] = nos_y
    hf_group["los"] = los
    hf_group["los at comp"] = los_y
    hf_group["tfdt"] = tfdt
    hf_group["tfdt max"] = tfdt_max
    hf_group["tfdt min"] = tfdt_min
    hf_group["trdt"] = trdt["all"]
    hf_group["trdt max"] = trdt_max
    hf_group["trdt min"] = trdt_min

    return bpp_cs, ba_cs, cost_cs, nos_cs, los_cs


def plot_city(city, save, modes=None, paths=None, params=None):
    """
    Plots the results for the given city. If no 'paths' or 'params' are given
    the default ones are created and used.

    :param city: Name of the city
    :type city: str
    :param save: Save name of the city.
    :type save: str
    :param modes:
    :type modes:
    :param paths: Dict with the paths for data, plots etc., check 'paths.py'
    in the example folder.
    :type paths: dict
    :param params: Dict with the params for plots etc., check 'params.py' in
    the example folder.
    :type params: dict
    :return: None
    """
    if modes is None:
        modes = [(False, 1, False)]
    if paths is None:
        paths = create_default_paths()
    if params is None:
        params = create_default_params()

    # Define city specific folders
    comp_folder = paths["comp_folder"]
    plot_folder = join(paths["plot_folder"], "results", save)
    input_folder = join(paths["input_folder"], save)
    output_folder = join(paths["output_folder"], save)

    # Create non existing folders
    Path(comp_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    hf_comp = h5py.File(join(comp_folder, f"comp_{save}.hdf5"), "w")
    hf_comp.attrs["city"] = city

    plt.rcdefaults()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    with open(join(input_folder, f"{save}_demand.json")) as demand_file:
        demand = json.load(demand_file)
    trip_nbrs = {literal_eval(k): v for k, v in demand.items()}
    trip_nbrs_re = {
        trip_id: nbr_of_trips
        for trip_id, nbr_of_trips in trip_nbrs.items()
        if not trip_id[0] == trip_id[1]
    }
    trips = sum([t for trip in trip_nbrs.values() for t in trip.values()])
    trips_re = sum([t for trip in trip_nbrs_re.values() for t in trip.values()])
    hf_comp.attrs["total trips"] = trips_re
    hf_comp.attrs["total trips (incl round trips)"] = trips
    utrips = len(trip_nbrs.keys())
    utrips_re = len(trip_nbrs_re.keys())
    hf_comp.attrs["unique trips"] = utrips_re
    hf_comp.attrs["unique trips (incl round trips)"] = utrips

    stations = [
        station for trip_id, nbr_of_trips in trip_nbrs.items() for station in trip_id
    ]
    stations = list(set(stations))
    hf_comp.attrs["nbr of stations"] = len(stations)
    hf_comp["stations"] = stations

    G = ox.load_graphml(
        filepath=join(input_folder, f"{save}.graphml"),
        node_dtypes={"street_count": float},
    )
    hf_comp.attrs["nodes"] = len(G.nodes)
    hf_comp.attrs["edges"] = len(G.edges)

    print(
        f"City: {city}, "
        f"nodes: {len(G.nodes)}, edges: {len(G.edges)}, "
        f"stations: {len(stations)}, "
        f"unique trips: {utrips_re} (rt incl.: {utrips}), "
        f"trips: {trips_re} (rt incl.: {trips})"
    )

    st_ratio = get_street_type_ratio(G)
    hf_comp["ratio street type"] = json.dumps(st_ratio)
    sn_ratio = len(stations) / len(G.nodes())
    hf_comp["ratio stations nodes"] = sn_ratio

    if paths["use_base_polygon"]:
        base_save = save.split(paths["save_devider"])[0]
        polygon_path = join(paths["polygon_folder"], f"{base_save}.json")
        polygon = get_polygon_from_json(polygon_path)
    else:
        polygon_path = join(paths["polygon_folder"], f"{save}.json")
        polygon = get_polygon_from_json(polygon_path)

    remove_area = None
    if params["correct_area"]:
        if paths["use_base_polygon"]:
            base_save = save.split(paths["save_devider"])[0]
            correct_area_path = join(
                paths["polygon_folder"], f"{base_save}_delete.json"
            )
        else:
            correct_area_path = join(paths["polygon_folder"], f"{save}_delete.json")
        if Path(correct_area_path).exists():
            remove_area = get_polygons_from_json(correct_area_path)
        else:
            print("No polygons for area size correction found.")

    area = calc_polygon_area(polygon, remove_area)
    hf_comp.attrs["area"] = area
    print(f"Area {round(area, 1)}")
    sa_ratio = len(stations) / area
    hf_comp.attrs["ratio stations area"] = sa_ratio

    plot_used_nodes(
        city=city,
        save=save,
        G=G,
        trip_nbrs=trip_nbrs,
        stations=stations,
        plot_folder=plot_folder,
        params=params,
    )

    data = {}
    cs_bike_paths = []
    for m in modes:
        hf_in = h5py.File(
            join(output_folder, f"{save}_data_mode_{m[0]:d}{m[1]}{m[2]:d}.hdf5"),
            "r",
        )
        data[m] = hf_in["all"]
        if len(cs_bike_paths) < len(json.loads(data[m]["used_ps_edges"][()])):
            cs_bike_paths = json.loads(data[m]["used_ps_edges"][()])
    data_base = calc_comparison_state(
        save,
        "base",
        base_state=True,
        params=deepcopy(params),
        paths=paths,
    )
    data_opt = calc_comparison_state(
        save,
        "opt",
        opt_state=True,
        params=deepcopy(params),
        paths=paths,
    )
    data_ex_inf = calc_comparison_state(
        save,
        "ex_inf",
        bike_paths=get_ex_inf(G),
        ex_inf=False,
        params=deepcopy(params),
        paths=paths,
    )
    data_cs = calc_comparison_state(
        save,
        "cs",
        bike_paths=cs_bike_paths,
        ex_inf=False,
        params=deepcopy(params),
        paths=paths,
    )
    data_cs_ex_inf = calc_comparison_state(
        save,
        "cs_ex_inf",
        bike_paths=cs_bike_paths,
        ex_inf=True,
        params=deepcopy(params),
        paths=paths,
    )
    if params["prerun_comp_states"]:
        print("Stopping plot after comp preparation.")
        return 0

    grp_algo = hf_comp.create_group("algorithm")
    for m, d in data.items():
        mode_str = f"{m[0]:d}{m[1]}{m[2]:d}"
        if params["cut"]:
            end_plot = d["cut"][()]
        else:
            end_plot = len(d["bpp"][()])
        print(f"Mode {mode_str}: cut after {end_plot}")

        sbgrp_algo = grp_algo.create_group(mode_str)
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
            stations=stations,
            mode=m,
            end_plot=end_plot,
            evo=evo,
            hf_group=sbgrp_algo,
            plot_folder=plot_folder,
            params=params,
            paths=paths,
        )
        if m[2]:
            if "p+s ex_inf" not in hf_comp.keys():
                grp_ps = hf_comp.create_group("p+s ex_inf")
                grp_ps["bpp"] = bpp_cs
                grp_ps["ba"] = ba_cs
                grp_ps["cost"] = cost_cs
                grp_ps["nos"] = nos_cs
                grp_ps["los"] = los_cs
        else:
            if "cs" not in hf_comp.keys():
                grp_ps = hf_comp.create_group("cs")
                grp_ps["bpp"] = bpp_cs
                grp_ps["ba"] = ba_cs
                grp_ps["cost"] = cost_cs
                grp_ps["nos"] = nos_cs
                grp_ps["los"] = los_cs
    hf_comp.close()


def comp_city(city, base_save, factors, mode, params, paths, labels=None):
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
        data = h5py.File(join(comp_folder, f"comp_{save}.hdf5"), "r")
        data_grp = data["algorithm"][mode_save]
        bpps[factor] = data_grp["bpp complete"][()]
        bpps_end[factor] = data_grp["bpp at end"][()]
        bas[factor] = data_grp["ba"][()]
        ends[factor] = data_grp["end"][()]
        data.close()

    end = max(ends.values())

    for factor in factors:
        bpps[factor] = [x / bpps[factor][end] for x in bpps[factor][: end + 1]]

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)

    ax.set_ylabel("bikeability b($\lambda$)", fontsize=params["fs_axl"])
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

    cmap = plt.cm.get_cmap("viridis")

    colors = [cmap(i) for i in np.linspace(0, 1, num=len(factors) - 1)]
    colors.insert(0, "k")
    for idx, factor in enumerate(factors):
        ax.plot(
            bpps[factor],
            bas[factor][: end + 1],
            label=labels[factor],
            c=colors[idx],
            lw=params["lw_ba"],
        )

    if params["titles"]:
        ax.set_title("Bikeability", fontsize=params["fs_title"])
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


def comp_modes(city, save, modes, params, paths):
    comp_folder = paths["comp_folder"]

    bpps = dict()
    bas = dict()
    costs = dict()
    ends = dict()

    for mode in modes:
        mode_save = f"{mode[0]:d}{mode[1]}{mode[2]:d}"
        data = h5py.File(join(comp_folder, f"comp_{save}.hdf5"), "r")
        data_grp = data["algorithm"][mode_save]
        bpps[mode_save] = data_grp["bpp complete"][()]
        bas[mode_save] = data_grp["ba"][()]
        costs[mode_save] = data_grp["cost complete"][()]
        ends[mode_save] = data_grp["end"][()]
        data.close()

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

    ax.set_ylabel("bikeability b($\lambda$)", fontsize=params["fs_axl"])
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
        ax.set_title("Bikeability", fontsize=params["fs_title"])
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


def plot_city_comparison(cities, paths, params, figsave="comp_cities", mode="010"):
    """

    :param cities:
    :param paths:
    :param params:
    :type params: dict
    :param figsave:
    :param mode:
    :return:
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
        print(join(comp_folder, f"comp_{save}.hdf5"))
        data_city = h5py.File(join(comp_folder, f"comp_{save}.hdf5"), "r")
        bpp[city] = data_city["algorithm"][mode]["bpp normed"][()]
        ba[city] = data_city["algorithm"][mode]["ba"][()]
        data_city.close()

    fig1, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    if isinstance(params["cmap_city_comp"], ListedColormap):
        cmap = params["cmap_city_comp"]
    elif isinstance(params["cmap_city_comp"], str):
        cmap = plt.cm.get_cmap(params["cmap_city_comp"])
    else:
        print(
            "Colormap must be colormap obbject or string. "
            "Defaulting to viridis colormap."
        )
        cmap = plt.cm.get_cmap("viridis")

    for idx, (city, save) in enumerate(cities.items()):
        ax1.plot(
            bpp[city],
            ba[city],
            color=cmap(idx / len(cities)),
            label=f"{city}",
            lw=params["lw_ba"],
        )

    ax1.set_ylabel("bikeability b($\lambda$)", fontsize=params["fs_axl"])
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


def plot_ex_inf_comparison(city, save, paths, params):
    plt.rcdefaults()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    comp_folder = paths["comp_folder"]

    data_city = h5py.File(join(comp_folder, f"comp_{save}.hdf5"), "r")
    bpp = data_city["algorithm"]["010"]["bpp"][()]
    ba = data_city["algorithm"]["010"]["ba"][()]
    bpp_cs = data_city["cs"]["bpp"][()]
    ba_cs = data_city["cs"]["ba"][()]
    bpp_exinf = data_city["algorithm"]["011"]["bpp"][()]
    ba_exinf = data_city["algorithm"]["011"]["ba"][()]
    bpp_exinf_cs = data_city["p+s ex_inf"]["bpp"][()]
    ba_exinf_cs = data_city["p+s ex_inf"]["ba"][()]
    data_city.close()

    fig1, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    ax1.plot(bpp, ba, label=f"{city}", c="C0", lw=params["lw_ba"])
    ax1.plot(bpp_cs, ba_cs, c="C0", ms=params["ms_ba"], marker=params["m_ba"])
    ax1.plot(
        bpp_exinf,
        ba_exinf,
        label=f"{city} with ex_inf",
        c="C1",
        lw=params["lw_ba"],
    )
    ax1.plot(
        bpp_exinf_cs,
        ba_exinf_cs,
        c="C1",
        ms=params["ms_ba"],
        marker=params["m_ba"],
    )
    ymax = ba_exinf[0]
    xmax = bpp_exinf[0]
    ax1.axvline(
        x=xmax,
        ymax=ymax,
        ymin=0,
        c=params["c_cost"],
        ls="--",
        alpha=0.5,
        lw=params["lw_cost"],
    )
    ax1.axhline(
        y=ymax,
        xmax=xmax,
        xmin=0,
        c=params["c_cost"],
        ls="--",
        alpha=0.5,
        lw=params["lw_ba"],
    )

    ax1.set_ylabel("bikeability b($\lambda$)", fontsize=params["fs_axl"])
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

    ax1.legend(loc="lower right", fontsize=params["fs_legend"], frameon=False)

    fig1.savefig(
        join(
            paths["plot_folder"],
            "results",
            save,
            f'{save}_comp_exinf.{params["plot_format"]}',
        ),
        bbox_inches="tight",
    )
