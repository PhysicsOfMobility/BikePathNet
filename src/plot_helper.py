"""
This module includes all necessary helper functions for the plotting
functionality.
"""
import h5py
import json
import pyproj
import numpy as np
import networkx as nx
import osmnx as ox
import shapely.ops as ops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ast import literal_eval
from math import ceil, floor, log10, radians
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import to_rgba, LogNorm
from functools import partial
from copy import deepcopy
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances
from .data_helper import get_street_type, get_street_length


def binding(x, kd, bmax):
    return (bmax * x) / (x + kd)


def logistic(x, x0, L, k):
    return L / (1 + np.exp(-k * (x - x0)))


def load_algorithm_params(path, params):
    f = open(path)
    data = json.load(f)

    params["street_cost"] = data["street_cost"]
    params["car_penalty"] = data["car_penalty"]
    params["slope_penalty"] = data["slope_penalty"]
    params["turn_penalty"] = data["turn_penalty"]

    return params


def magnitude(x):
    """
    Calculate the magnitude of x.
    :param x: Number to calculate the magnitude of.
    :type x: numeric (e.g. float or int)
    :return: Magnitude of x
    :rtype: int
    """
    if x != 0.0:
        return int(floor(log10(x)))
    else:
        return 0


def get_ex_inf(G):
    ex_inf = list()
    for u, v, data in G.edges(data=True):
        if not isinstance(data["ex_inf"], bool):
            data["ex_inf"] = literal_eval(data["ex_inf"])
        if data["ex_inf"]:
            ex_inf.append((u, v))
    return ex_inf


def total_len_of_bike_paths(G):
    total_len = 0
    for e0, e1, d in G.edges(data=True):
        if not isinstance(d["ramp"], bool):
            d["ramp"] = literal_eval(d["ramp"])
        if not d["ramp"]:
            total_len += d["length"]
    return total_len


def fraction_of_bike_paths(edited_edges, G):
    total_len = 0
    bike_path_len = 0
    for e0, e1, d in G.edges(data=True):
        if not isinstance(d["ramp"], bool):
            d["ramp"] = literal_eval(d["ramp"])
        if not d["ramp"]:
            total_len += d["length"]
            if (e0, e1, 0) in edited_edges:
                bike_path_len += d["length"]
    if total_len == 0:
        return 0
    else:
        return bike_path_len / total_len


def len_of_bikepath_by_type(ee, G, buildup):
    """
    Calculates the length of bike paths along the different street types.
    :param ee: List of edited edges.
    :type ee: list
    :param G: Street graph.
    :type G: networkx graph
    :param buildup: Reversed algorithm used True/False
    :type buildup: bool
    :return: Dictionary keyed by street type
    :rtype: dict
    """
    street_types = ["primary", "secondary", "tertiary", "residential"]
    total_len = {k: 0 for k in street_types}
    for e in G.edges():
        st = get_street_type(G, e)
        total_len[st] += G[e[0]][e[1]]["length"]
    len_fraction = {k: [0] for k in street_types}
    if not buildup:
        ee = list(reversed(ee))
    for e in ee:
        st = get_street_type(G, e)
        len_before = len_fraction[st][-1]
        len_fraction[st].append(len_before + G[e[0]][e[1]]["length"] / total_len[st])
        for s in [s for s in street_types if s != st]:
            len_fraction[s].append(len_fraction[s][-1])
    return len_fraction


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


def total_distance_traveled_list(total_dist, total_dist_cs, buildup):
    """
    Renormalises all total distance traveled lists.
    :param total_dist: dict of tdt lists
    :type total_dist: dict
    :param total_dist_cs:  dict of tdt lists for the comparison state
    :type total_dist_cs: dict
    :param buildup: Reversed algorithm used True/False
    :type buildup: bool
    :return:
    """

    dist = {}
    dist_cs = {}

    # On all
    on_all = [i["total length on all"] for i in total_dist]
    on_all_cs = total_dist_cs["total length on all"]

    dist["all"] = on_all
    dist_cs["all"] = on_all_cs

    # On streets w/o bike paths
    on_street = [i["total length on street"] for i in total_dist]
    dist["street"] = [x / on_all[idx] for idx, x in enumerate(on_street)]
    dist_cs["street"] = total_dist_cs["total length on street"] / on_all_cs

    # On primary
    on_primary = [i["total length on primary"] for i in total_dist]
    dist["primary"] = [x / on_all[idx] for idx, x in enumerate(on_primary)]
    dist_cs["primary"] = total_dist_cs["total length on primary"] / on_all_cs

    # On secondary
    on_secondary = [i["total length on secondary"] for i in total_dist]
    dist["secondary"] = [x / on_all[idx] for idx, x in enumerate(on_secondary)]
    dist_cs["secondary"] = total_dist_cs["total length on secondary"] / on_all_cs
    # On tertiary
    on_tertiary = [i["total length on tertiary"] for i in total_dist]
    dist["tertiary"] = [x / on_all[idx] for idx, x in enumerate(on_tertiary)]
    dist_cs["tertiary"] = total_dist_cs["total length on tertiary"] / on_all_cs

    # On residential
    on_residential = [i["total length on residential"] for i in total_dist]
    dist["residential"] = [x / on_all[idx] for idx, x in enumerate(on_residential)]
    dist_cs["residential"] = total_dist_cs["total length on residential"] / on_all_cs

    # On bike paths
    on_bike = [i["total length on bike paths"] for i in total_dist]
    dist["bike paths"] = [x / on_all[idx] for idx, x in enumerate(on_bike)]
    dist_cs["bike paths"] = total_dist_cs["total length on bike paths"] / on_all_cs

    if not buildup:
        for st, len_on_st in dist.items():
            dist[st] = list(reversed(len_on_st))
    return dist, dist_cs


def sum_total_cost(cost, buildup):
    """
    Sums up all total cost up to each step.
    :param cost: List of costs per step
    :type cost: list
    :param cost_cs: Cost of the comparison state.
    :type cost_cs: float  or int
    :param buildup: Reversed algorithm used True/False
    :type buildup: bool
    :return: Summed cost
    :rtype: list
    """
    if not buildup:
        cost = list(reversed(cost))  # costs per step
    total_cost = [sum(cost[:i]) for i in range(0, len(cost))]
    # total_cost = [i / total_cost[-1] for i in total_cost]
    return total_cost


def get_end(tdt, tdt_opt, tdt_base, buildup, felt=True):
    """
    Returns the index where the bikeability reaches 1.
    :param tdt: total distance traveled
    :type tdt: dict
    :param tdt_opt: total distance traveled for optimal state
    :type tdt_opt: dict
    :param tdt_base: total distance traveled for base state
    :type tdt_base: dict
    :param buildup: Reversed algorithm used True/False
    :type buildup: bool
    :param felt: If felt distance
    :type felt: bool
    :return: Index where bikeability reaches 1
    :rtype: int
    """
    if felt:
        tdt_l = tdt
        tdt_base_l = tdt_base
        tdt_opt_l = tdt_opt
    else:
        tdt_l, tdt_opt_l = total_distance_traveled_list(tdt, tdt_opt, buildup)
        tdt_l, tdt_base_l = total_distance_traveled_list(tdt, tdt_base, buildup)
        tdt_l = tdt_l["all"]
        tdt_base_l = tdt_base_l["all"]
        tdt_opt_l = tdt_opt_l["all"]
    tdt_min = tdt_opt_l
    tdt_max = tdt_base_l
    ba = [(tdt_max - i) / (tdt_max - tdt_min) for i in tdt_l]
    if not buildup:
        ba = list(reversed(ba))
    ba_max = min(1.0, max(ba))
    return next(x for x, val in enumerate(ba) if val >= ba_max)


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
        e_st = get_street_type(G, edge)
        e_len = get_street_length(G, edge)
        st_len[e_st] += e_len
        total_len += e_len
    st_len_norm = {k: v / total_len for k, v in st_len.items()}
    return st_len_norm


def calc_polygon_area(polygon, remove=None, unit="sqkm"):
    """
    Calculates the area of a given lat/long Polygon.
    :param polygon: Polygon to caculate the area of
    :type polygon: shapely polygon
    :param remove: Polygons inside the orignal polygon to exclude from the
    area calculation
    :type remove: list of shapely polygons
    :param unit: Unit in which the area is returned km^2 = 'sqkm' or m^2 =
    'sqm'
    :type unit: str
    :return:
    """
    if (not isinstance(remove, list)) ^ (remove is None):
        remove = [remove]
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init="EPSG:4326"),
            pyproj.Proj(proj="aea", lat_1=polygon.bounds[1], lat_2=polygon.bounds[3]),
        ),
        polygon,
    )
    remove_area = 0
    if remove is not None:
        for p in remove:
            a_r = ops.transform(
                partial(
                    pyproj.transform,
                    pyproj.Proj("EPSG:4326"),
                    pyproj.Proj(proj="aea", lat_1=p.bounds[1], lat_2=p.bounds[3]),
                ),
                p,
            )
            remove_area += a_r.area

    if unit == "sqkm":
        return (geom_area.area - remove_area) / 1000000
    if unit == "sqm":
        return geom_area.area - remove_area


def calc_scale(base_city, cities, saves, comp_folder, mode):
    """
    Calculates the x scaling for city comparison.
    :param base_city: Base city of the caluclation
    :type base_city: str
    :param cities: List of cities to calculate
    :type cities: list
    :param saves: Dictionary mapping cities to save abbreviations
    :type saves: dict
    :param comp_folder: Path to the folder where the comparison data is stored
    :type comp_folder: str
    :param mode: Mode of the simulation
    :type mode: str
    :return: Dictionary of scale factors keyed by city
    :rtype: dict
    """
    blp = {}
    ba = {}

    if isinstance(mode, tuple):
        mode = f"{mode[0]:d}{mode[1]}"

    for city in cities:
        save = saves[city]
        data = h5py.File(f"{comp_folder}comp_{save}.hdf5", "r")
        blp[city] = data["algorithm"][mode]["bpp"][()]
        ba[city] = data["algorithm"][mode]["ba"][()]

    blp_base = blp[base_city]
    ba_base = ba[base_city]

    cities_comp = deepcopy(cities)
    cities_comp.remove(base_city)

    min_idx = {}
    for city in cities_comp:
        m_idx = []
        for idx, x in enumerate(ba[city]):
            # Create list where each value corresponds to the index of the
            # item from the base city ba list closest to the ba value of the
            # comparing city at the current index.
            m_idx.append(min(range(len(ba_base)), key=lambda i: abs(ba_base[i] - x)))
        min_idx[city] = m_idx

    scale = {}
    for city in cities_comp:
        scale_city = []
        min_idx_city = min_idx[city]
        blp_city = blp[city]
        for idx, x in enumerate(min_idx_city):
            if blp_city[idx] != 0:
                scale_city.append(blp_base[x] / blp_city[idx])
            else:
                scale_city.append(np.nan)
        scale[city] = scale_city

    scale_mean = {}
    for city in cities:
        if city == base_city:
            scale_mean[city] = 1.0
        else:
            scale_mean[city] = np.mean(scale[city][1:])
    return scale_mean


def get_edge_color_bp(G, edges, color, ex_inf, color_ex_inf, color_unused):
    ec = []
    for u, v, d in G.edges(data=True):
        if ex_inf and d["ex_inf"]:
            ec_uv = color_ex_inf
        elif [u, v] in edges:
            ec_uv = color
        else:
            ec_uv = color_unused
        ec.append(ec_uv)

    return ec


def get_edge_color(G, edges, attr, color):
    """
    Return edge color list for G, edges have the given color if they are
    part of the edges list and therefore hav the given attribute, otherwise
    they have the color '#999999'.
    :param G: Graph
    :type G: networkx graph
    :param edges: List of edges which have the attribute
    :type edges: list
    :param attr: Attribute for the coloring (e.g. bike path)
    :type attr: int, float or str
    :param color: Color ift edge has attribute
    :type color: color (e.g. hexcode)
    :return: List of edge colors for graph G.
    :rtype: list
    """
    nx.set_edge_attributes(G, False, attr)
    for edge in edges:
        G[edge[0]][edge[1]][0][attr] = True
        G[edge[1]][edge[0]][0][attr] = True
    return [
        color if data[attr] else "#999999"
        for u, v, data in G.edges(keys=False, data=True)
    ]


def get_edge_color_st(G, colors):
    """
     Return edge color list, to color the edges depending on their street
     type.
    :param G: Graph
    :type G: osmnx graph
    :param colors: Dictionary for the street type colors
    :type colors: dict
    :return: List of edge colors for graph G.
    :rtype: list
    """
    return [colors[get_street_type(G, e)] for e in G.edges()]


def plot_barh(
    data,
    colors,
    save,
    figsize=None,
    x_label="",
    title="",
    dpi=150,
):
    """
    Plot a horizontal bar plot.
    :param data: Data to plot as dictionary.
    :type data: dict
    :param colors: Colors for the data
    :type colors: dict
    :param save: Save location
    :type save: str
    :param figsize: Size of figure (width, height)
    :type figsize: tuple
    :param x_label: Label for the x axis
    :type x_label: str
    :param title: Title of the plot
    :type title: str
    :param dpi: dpi of the plot
    :type dpi: int
    :return: None
    """
    if figsize is None:
        figsize = [16, 9]
    keys = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    y_pos = np.arange(len(keys))
    max_value = max(values)
    for idx, key in enumerate(keys):
        color = to_rgba(colors[key])
        ax.barh(y_pos[idx], values[idx], color=color, align="center")
        x = values[idx] / 2
        y = y_pos[idx]
        if values[idx] > 0.05 * max_value:
            r, g, b, _ = color
            text_color = (
                "white" if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25 else "black"
            )
            ax.text(
                x,
                y,
                f"{values[idx]:3.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=16,
            )
        else:
            ax.text(
                2 * values[idx],
                y,
                f"{values[idx]:3.2f}",
                ha="center",
                va="center",
                color="darkgrey",
                fontsize=16,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(x_label)
    ax.set_title(title)

    plt.savefig(save, bbox_inches="tight")


def plot_barh_stacked(
    data,
    stacks,
    colors,
    save,
    figsize=None,
    title="",
    dpi=150,
    legend=False,
):
    """
    Plot a stacked horizontal bar plot.
    :param data: Data to plot as dictionary.
    :type data: dict
    :param colors: Colors for the data
    :type colors: list
    :param save: Save location
    :type save: str
    :param figsize: Size of figure (width, height)
    :type figsize: tuple
    :param title: Title of the plot
    :type title: str
    :param dpi: dpi of the plot
    :type dpi: int
    :param legend: If legend should be plotted or not
    :type legend: bool
    :return: None
    """
    if figsize is None:
        figsize = [16, 9]

    labels = list(data.keys())
    values = np.array(list(data.values()))
    values_cum = values.cumsum(axis=1)
    colors = [to_rgba(c) for c in colors]

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, max(np.sum(values, axis=1)))

    for i, (colname, color) in enumerate(zip(stacks, colors)):
        widths = values[:, i]
        starts = values_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = "white" if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25 else "black"
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c != 0.0:
                ax.text(
                    x,
                    y,
                    f"{c:3.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )
    if legend:
        ax.legend(
            ncol=len(stacks),
            bbox_to_anchor=(0, 1),
            loc="lower left",
            fontsize="small",
        )
    ax.set_title(title)
    plt.savefig(save, bbox_inches="tight")


def plot_barv(
    data,
    colors,
    save,
    figsize=None,
    y_label="",
    title="",
    ymin=-0.1,
    ymax=0.7,
    xticks=True,
    dpi=150,
):
    """
    Plot a vertical bar plot.
    :param data: Data to plot as dictionary.
    :type data: dict
    :param colors: Colors for the data
    :type colors: dict
    :param save: Save location
    :type save: str
    :param figsize: Size of figure (width, height)
    :type figsize: tuple
    :param y_label: Label for the x axis
    :type y_label: str
    :param title: Title of the plot
    :type title: str
    :param ymin: Minimal y value for axis
    :type ymin: float
    :param ymax: Maximal y value for axis
    :type ymax: float
    :param xticks: Plot x ticks or not
    :type xticks: bool
    :param dpi: dpi of the plot
    :type dpi: int
    :return: None
    """
    if figsize is None:
        figsize = [10, 10]
    keys = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.set_ylim(ymin, ymax)
    x_pos = np.arange(len(keys))
    for idx, key in enumerate(keys):
        color = to_rgba(colors[key])
        ax.bar(x_pos[idx], values[idx], color=color, align="center")
        y = values[idx] / 2
        x = x_pos[idx]
        r, g, b, _ = color
        text_color = "white" if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25 else "black"
        ax.text(
            x,
            y,
            f"{values[idx]:3.2f}",
            ha="center",
            va="center",
            color=text_color,
        )
    if xticks:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(keys)
        ax.tick_params(axis="x", labelsize=24)
    else:
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_xlabel("", fontsize=24)
    ax.set_title(title)

    plt.savefig(save, bbox_inches="tight")


def plot_barv_stacked(
    labels,
    data,
    colors,
    title="",
    ylabel="",
    save="",
    width=0.8,
    figsize=None,
    dpi=150,
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
    :param save: Save location
    :type save: str
    :param width: Width of bars
    :type width: float
    :param figsize: Size of figure (width, height)
    :type figsize: tuple
    :param dpi: dpi of the plot
    :type dpi: int
    :return:
    """
    if figsize is None:
        figsize = [10, 12]

    stacks = list(data.keys())
    values = list(data.values())
    x_pos = np.arange((len(labels)))
    x_pos = [x / 2 for x in x_pos]

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
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

    ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(title, fontsize=12)

    plt.savefig(save, bbox_inches="tight")


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
    dpi=150,
    figsize=(4, 4),
):
    """
    Plot a histogram
    :param data:
    :param save_path:
    :param bins:
    :param cumulative:
    :param density:
    :param xlabel:
    :param ylabel:
    :param xlim:
    :param xaxis:
    :param xticks:
    :param cm:
    :param dpi:
    :param figsize:
    :return:
    """
    max_d = max(data)
    min_d = min(data)
    r = magnitude(max_d)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
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
        ax.hist(
            data,
            bins=bins,
            align="mid",
            cumulative=cumulative,
            density=density,
        )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
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

    fig.savefig(save_path, bbox_inches="tight")


def intensity_decay(distance):
    if isinstance(distance, float):
        scale = distance / 10
    else:
        scale = distance.max() / 10

    return (1 / np.sqrt(2 * np.pi * (scale**2))) * np.exp(
        -0.5 * (distance / scale) ** 2
    )


def calc_intensity(x, y, station_location, station_usage, cutoff):
    grid_coord = np.array([(radians(y_i), radians(x_i)) for y_i in y for x_i in x])

    stations = list(station_location.keys())
    usage = np.array([station_usage[station] for station in stations])

    stations_coord = np.array(
        [
            (
                radians(station_location[station][1]),
                radians(station_location[station][0]),
            )
            for station in stations
        ]
    )
    distances = haversine_distances(grid_coord, stations_coord) * 6371000

    intensity = 2 * norm.pdf(distances, scale=0.3 * cutoff)
    intensity[intensity < 2 * norm.pdf(cutoff, scale=0.3 * cutoff)] = 0
    intensity = usage * intensity

    return np.sum(intensity, axis=1)


def plot_node_usage_bg(G, node_loads, ax, figsize, dpi, min_x, max_x, min_y, max_y):
    node_locations = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}

    edge_lengths = [d["length"] for u, v, d in G.edges(data=True)]
    """edge_lengths = [l for l in edge_lengths
                    if np.percentile(edge_lengths, 5) <= l <=
                    np.percentile(edge_lengths, 95)]"""

    cutoff = np.mean(edge_lengths)

    x = np.linspace(min_x, max_x, int(figsize[0] * dpi))
    y = np.linspace(min_y, max_y, int(figsize[1] * dpi))
    z = calc_intensity(x, y, node_locations, node_loads, cutoff)

    X, Y = np.meshgrid(x, y)
    Z = z.reshape(len(y), len(x))

    ax.contourf(X, Y, Z, cmap="magma", zorder=0, levels=100)
