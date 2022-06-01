"""
This module includes all necessary helper functions for the plotting
functionality.
"""
import pyproj
import numpy as np
import networkx as nx
import shapely.ops as ops
from math import ceil, floor, log10
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import to_rgba
from functools import partial
from .algorithm_helper import get_street_type_cleaned, get_street_length
from .setup_helper import create_default_params, create_default_paths


def magnitude(x):
    """
    Calculate the magnitude of x.
    :param x: Number to calculate the magnitude of.
    :type x: numeric (e.g. float or int)
    :return: Magnitude of x
    :rtype: int
    """
    return int(floor(log10(x)))


def coord_transf(x, y, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    """
    Transfers the coordinates from data to relative coordiantes.
    :param x: x data coordinate
    :type x: float or int
    :param y: y data coordinate
    :type y: float or int
    :param xmin: min value of x axis
    :type xmin: float or int
    :param xmax: max value of x axis
    :type xmax: float or int
    :param ymin: min value of y axis
    :type ymin: float or int
    :param ymax: max value of y axis
    :type ymax: float or int
    :return: transformed x, y coordinates
    :rtype: float, float
    """
    return (x - xmin) / (xmax - xmin), (y - ymin) / (ymax - ymin)


def total_distance_traveled_list(total_dist, total_dist_now):
    """
    Renormalises all total distance traveled lists.
    :param total_dist: dict of tdt lists
    :type total_dist: dict
    :param total_dist_now:  dict of tdt lists for the current state
    :type total_dist_now: dict
    :return:
    """
    dist = {}
    dist_now = {}

    # On all
    on_all = [i["total length on all"] for i in total_dist]
    on_all_now = total_dist_now["total length on all"]

    dist["all"] = on_all
    dist_now["all"] = total_dist_now["total length on all"]

    # On streets w/o bike paths
    on_street = [i["total length on street"] for i in total_dist]
    dist["street"] = [x / on_all[idx] for idx, x in enumerate(on_street)]
    dist_now["street"] = total_dist_now["total length on street"] / on_all_now

    # On primary
    on_primary = [i["total length on primary"] for i in total_dist]
    dist["primary"] = [x / on_all[idx] for idx, x in enumerate(on_primary)]
    dist_now["primary"] = total_dist_now["total length on primary"] / on_all_now

    # On secondary
    on_secondary = [i["total length on secondary"] for i in total_dist]
    dist["secondary"] = [x / on_all[idx] for idx, x in enumerate(on_secondary)]
    dist_now["secondary"] = total_dist_now["total length on secondary"] / on_all_now
    # On tertiary
    on_tertiary = [i["total length on tertiary"] for i in total_dist]
    dist["tertiary"] = [x / on_all[idx] for idx, x in enumerate(on_tertiary)]
    dist_now["tertiary"] = total_dist_now["total length on tertiary"] / on_all_now

    # On residential
    on_residential = [i["total length on residential"] for i in total_dist]
    dist["residential"] = [x / on_all[idx] for idx, x in enumerate(on_residential)]
    dist_now["residential"] = total_dist_now["total length on residential"] / on_all_now

    # On bike paths
    on_bike = [i["total length on bike paths"] for i in total_dist]
    dist["bike paths"] = [x / on_all[idx] for idx, x in enumerate(on_bike)]
    dist_now["bike paths"] = total_dist_now["total length on bike paths"] / on_all_now

    for st, len_on_st in dist.items():
        dist[st] = list(reversed(len_on_st))
    return dist, dist_now


def sum_total_cost(cost, cost_now):
    """
    Sums up all total cost up to each step.
    :param cost: List of costs per step
    :type cost: list
    :param cost_now: Cost of the current state.
    :type cost_now: float  or int
    :return: Summed and renormalised cost and renormalised cost for the
    current state
    :rtype: list, float
    """
    cost = list(reversed(cost))  # costs per step
    total_cost = [sum(cost[:i]) for i in range(1, len(cost) + 1)]
    cost_now = cost_now / total_cost[-1]
    total_cost = [i / total_cost[-1] for i in total_cost]
    return total_cost, cost_now


def get_end(tdt, tdt_now):
    """
    Returns the index where the bikeability reaches 1.
    :param tdt: total distance traveled
    :type tdt: dict
    :param tdt_now: total distance traveled for current state
    :type tdt_now: dict
    :return: Index where bikeability reaches 1
    :rtype: int
    """
    tdt, tdt_now = total_distance_traveled_list(tdt, tdt_now)
    ba = [
        1 - (i - min(tdt["all"])) / (max(tdt["all"]) - min(tdt["all"]))
        for i in tdt["all"]
    ]
    return next(x for x, val in enumerate(ba) if val >= 1)


def get_street_type_ratio(G):
    """
    Gets the ratios for the different street types in the given graph.
    :param G: Street network.
    :type G: osmnx graph
    :return: Street type ratio in dict keyed by street type
    :rtype: dict
    """
    G = G.to_undirected()
    G = nx.Graph(G)
    st_len = {"primary": 0, "secondary": 0, "tertiary": 0, "residential": 0}
    total_len = 0
    for edge in G.edges:
        e_st = get_street_type_cleaned(G, edge, multi=False)
        e_len = get_street_length(G, edge, multi=False)
        st_len[e_st] += e_len
        total_len += e_len
    st_len_norm = {k: v / total_len for k, v in st_len.items()}
    return st_len_norm


def calc_polygon_area(polygon, unit="sqkm"):
    """
    Calculates the area of a given lat/long Polygon.
    :param polygon: Polygon to caculate the area of
    :type polygon: shapely polygon
    :param unit: Unit in which the area is returned km^2 = 'sqkm' or m^2 =
    'sqm'
    :type unit: str
    :return:
    """
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init="EPSG:4326"),
            pyproj.Proj(proj="aea", lat_1=polygon.bounds[1], lat_2=polygon.bounds[3]),
        ),
        polygon,
    )

    if unit == "sqkm":
        return geom_area.area / 1000000
    if unit == "sqm":
        return geom_area.area


def plot_barv_stacked(
    labels, data, colors, title="", ylabel="", save="", width=0.8, params=None
):
    """
    Plot a stacked vertical bar plot
    :param labels: Labels for the bars
    :type labels: list
    :param data: Data to plot as dictionary.
    :type data: dict
    :param colors: Colors for the data
    :type colors: dict
    :param title: Title of the plot
    :type title: str
    :param ylabel: Label for y axis
    :type ylabel: str
    :param save: Save location without format
    :type save: str
    :param width: Width of bars
    :type width: float
    :return:
    """
    if params is None:
        figsize = [10, 12]

    stacks = list(data.keys())
    values = list(data.values())
    x_pos = np.arange((len(labels)))
    x_pos = [x / 2 for x in x_pos]

    fig, ax = plt.subplots(figsize=params["figs_comp_st"], dpi=params["dpi"])
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.5)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    bottom = np.zeros(len(values[0]))
    for idx in range(len(stacks)):
        ax.bar(
            x_pos,
            values[idx],
            width,
            label=stacks[idx],
            bottom=bottom,
            color=colors[stacks[idx]],
        )
        for v_idx, v in enumerate(values[idx]):
            if v > 0.05:
                color = to_rgba(colors[stacks[idx]])
                y = bottom[v_idx] + v / 2
                x = x_pos[v_idx]
                r, g, b, _ = color
                text_color = (
                    "white" if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25 else "black"
                )
                ax.text(
                    x,
                    y,
                    f"{v:3.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=5,
                )
        bottom = [sum(x) for x in zip(bottom, values[idx])]
        # print(stacks[idx], values[idx])

    ax.set_ylabel(ylabel, fontsize=params["fs_axl"])
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="y", labelsize=params["fs_ticks"])
    ax.tick_params(axis="x", labelsize=params["fs_ticks"])
    ax.set_title(title, fontsize=params["fs_title"])

    plt.savefig(f'{save}.{params["plot_format"]}', bbox_inches="tight")


def plot_histogram(
    data,
    save_path,
    bins=None,
    cumulative=False,
    density=False,
    xlabel="",
    ylabel="",
    xlim=None,
    xaxis=True,
    xticks=None,
    cm=None,
    params=None,
):
    """
    Plot a histogram
    :param data: Data to plot
    :param save_path: Path to save to
    :param bins: Bins for the hist plot. See matplotlib hist()
    :param cumulative: plot cumulative hist
    :type cumulative: bool
    :param density: see matplotlib hist()
    :param xlabel: x label
    :type xlabel: str
    :param ylabel: y label
    :type ylabel: str
    :param xlim: Limit of the x axis
    :param xaxis: If to plot x axis or not.
    :type xaxis: bool
    :param xticks: ticks vor x-axis
    :type  xticks
    :param cm: Colour map for bins
    :param params: Params for plotting.
    :type params: dict
    :return: None
    """
    max_d = max(data)
    min_d = min(data)
    r = magnitude(max_d)

    fig, ax = plt.subplots(figsize=params["figs_station_usage_hist"], dpi=params["dpi"])
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=0.0, right=round(max_d + 0.1 * max_d, -(r - 1)))
    if bins is None:
        if max_d == min_d:
            bins = 1
        elif (max_d - min_d) <= 200:
            bins = 50
        else:
            bins = ceil((max_d - min_d) / (10 ** (r - 2)))
    if cm is not None:
        n, b, patches = ax.hist(
            data,
            bins=bins,
            align="mid",
            color="green",
            cumulative=cumulative,
            density=density,
        )
        for i, p in enumerate(patches):
            plt.setp(p, "facecolor", cm(i / bins))
    else:
        ax.hist(data, bins=bins, align="mid", cumulative=cumulative, density=density)
    ax.set_xlabel(xlabel, fontsize=params["fs_axl"])
    ax.set_ylabel(ylabel, fontsize=params["fs_axl"])
    ax.tick_params(axis="both", labelsize=params["fs_ticks"])
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if xticks is not None:
        ax.set_xticks(list(xticks.keys()))
        ax.set_xticklabels(list(xticks.values()))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if not xaxis:
        ax.tick_params(
            axis="x", hich="both", bottom=False, top=False, labelbottom=False
        )

    fig.savefig(f'{save_path}.{params["plot_format"]}', bbox_inches="tight")
