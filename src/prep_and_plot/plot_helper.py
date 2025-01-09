"""
This module includes all necessary helper functions for the plotting
functionality.
"""
import json
import networkx as nx
import numpy as np
import osmnx as ox
import pyproj
import shapely
import shapely.ops as ops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ast import literal_eval
from collections.abc import Iterable
from collections.abc import Sequence
from functools import partial
from math import ceil, floor, log10
from matplotlib.axes._axes import Axes
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from osmnx.plot import _config_ax, _save_and_show
from pathlib import Path
from .data_helper import get_street_type


def binding(x, kd: float, bmax: float):
    """Binding function/curve (also called Hill equation).

    Parameters
    ----------
    x : array_like
        X-values to calculate the binding function for.
    kd : numeric (e.g. float or int)
        
    bmax : numeric (e.g. float or int)
        

    Returns
    -------
    output : float | ndarray
        Returns binding function of x
    """
    return (bmax*x)/(x+kd)


def logistic(x, x0: float, L: float, k: float):
    """Logistic function/curve.

    Parameters
    ----------
    x : array_like
        
    x0 : numeric (e.g. float or int)
        
    L : numeric (e.g. float or int)
        
    k : numeric (e.g. float or int)
        

    Returns
    -------
    output : ndarray
        Returns logistic function of values x
    """
    return L/(1+np.exp(-k*(x-x0)))


def load_comparison_state_results(result_file: str) -> dict:
    """Loads the results for a comparison state from the given file.

    Parameters
    ----------
    result_file : str
        Path to the result file.

    Returns
    -------
    output : dict
        Result data loaded as dictionary.
    """
    with open(result_file, 'r') as f:
        data = json.load(f)

    data["edge_loads"] = {tuple(literal_eval(k)): v for k, v in data["edge_loads"].items()}
    return data


def magnitude(x: float) -> int:
    """Calculate the magnitude of x.

    Parameters
    ----------
    x : numeric (e.g. float or int)
        Number to calculate the magnitude of.

    Returns
    -------
    output : int

    """
    if x != 0.0:
        return int(floor(log10(abs(x))))
    else:
        return 0


def len_of_bikepath_by_type(
        G: nx.MultiGraph | nx.MultiDiGraph,
        ee: list,
        buildup: bool = False
    ) -> dict:
    """Calculates the length of bike paths along the different street types.

    Parameters
    ----------
    ee : list
        Edited edges
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    buildup : bool (Default value = False)
        Reversed algorithm used True/False

    Returns
    -------
    dict
        Dictionary keyed by street type

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
        len_fraction[st].append(
            len_before + G[e[0]][e[1]]["length"] / total_len[st]
        )
        for s in [s for s in street_types if s != st]:
            len_fraction[s].append(len_fraction[s][-1])
    return len_fraction


def coord_transf(
        x: float,
        y: float,
        xmin: float = 0.0,
        xmax: float = 1.0,
        ymin: float = 0.0,
        ymax: float = 1.0
    ) -> tuple[float, float]:
    """Transfers the coordinates from data to relative coordiantes.

    Parameters
    ----------
    x : float
        x data coordinate
    y : float
        y data coordinate
    xmin : float
        min value of x-axis (Default value = 0.0)
    xmax : float
        max value of x-axis (Default value = 1.0)
    ymin : float
        min value of y-axis (Default value = 0.0)
    ymax : float
        max value of y-axis (Default value = 1.0)

    Returns
    -------
    output : tuple[float, float]

    """
    return (x - xmin) / (xmax - xmin), (y - ymin) / (ymax - ymin)


def total_distance_traveled_list(
        total_dist: dict,
        total_dist_cs: dict,
        buildup: bool = False
    ) -> tuple[dict, dict]:
    """Renormalises all total distance traveled lists.

    Parameters
    ----------
    total_dist : dict
        Dict of tdt lists
    total_dist_cs : dict
        Dict of tdt lists for the comparison state
    buildup : bool
        Reversed algorithm used True/False

    Returns
    -------
    output : tuple[dict, dict]
        Renormalised total distances for the complete data and comparison point.

    """

    dist = {}
    dist_cs = {}

    # On all
    if isinstance(total_dist, list):
        on_all = [i["total_length_on_all"] for i in total_dist]
        on_all_cs = total_dist_cs["total_length_on_all"]

        dist["all"] = on_all
        dist_cs["all"] = on_all_cs

        # On streets w/o bike paths
        on_street = [i["total_length_on_street"] for i in total_dist]
        dist["street"] = [x / on_all[idx] for idx, x in enumerate(on_street)]
        dist_cs["street"] = total_dist_cs["total_length_on_street"] / on_all_cs

        # On primary
        on_primary = [i["total_length_on_primary"] for i in total_dist]
        dist["primary"] = [x / on_all[idx] for idx, x in enumerate(on_primary)]
        dist_cs["primary"] = total_dist_cs["total_length_on_primary"] / on_all_cs

        # On secondary
        on_secondary = [i["total_length_on_secondary"] for i in total_dist]
        dist["secondary"] = [x / on_all[idx] for idx, x in enumerate(on_secondary)]
        dist_cs["secondary"] = (
            total_dist_cs["total_length_on_secondary"] / on_all_cs
        )
        # On tertiary
        on_tertiary = [i["total_length_on_tertiary"] for i in total_dist]
        dist["tertiary"] = [x / on_all[idx] for idx, x in enumerate(on_tertiary)]
        dist_cs["tertiary"] = total_dist_cs["total_length_on_tertiary"] / on_all_cs

        # On residential
        on_residential = [i["total_length_on_residential"] for i in total_dist]
        dist["residential"] = [
            x / on_all[idx] for idx, x in enumerate(on_residential)
        ]
        dist_cs["residential"] = (
            total_dist_cs["total_length_on_residential"] / on_all_cs
        )

        # On bike paths
        on_bike = [i["total_length_on_bike_paths"] for i in total_dist]
        dist["bike paths"] = [x / on_all[idx] for idx, x in enumerate(on_bike)]
        dist_cs["bike paths"] = (
            total_dist_cs["total_length_on_bike_paths"] / on_all_cs
        )
    else:
        on_all = total_dist["on_all"]
        on_all_cs = total_dist_cs["on_all"]

        dist["all"] = on_all
        dist_cs["all"] = on_all_cs

        # On streets w/o bike paths
        on_street = total_dist["on_street"]
        dist["street"] = [x / on_all[idx] for idx, x in enumerate(on_street)]
        dist_cs["street"] = total_dist_cs["on_street"] / on_all_cs

        # On primary
        on_primary = total_dist["on_primary"]
        dist["primary"] = [x / on_all[idx] for idx, x in enumerate(on_primary)]
        dist_cs["primary"] = total_dist_cs["on_primary"] / on_all_cs

        # On secondary
        on_secondary = total_dist["on_secondary"]
        dist["secondary"] = [x / on_all[idx] for idx, x in enumerate(on_secondary)]
        dist_cs["secondary"] = (
                total_dist_cs["on_secondary"] / on_all_cs
        )
        # On tertiary
        on_tertiary = total_dist["on_tertiary"]
        dist["tertiary"] = [x / on_all[idx] for idx, x in enumerate(on_tertiary)]
        dist_cs["tertiary"] = total_dist_cs["on_tertiary"] / on_all_cs

        # On residential
        on_residential = total_dist["on_residential"]
        dist["residential"] = [
            x / on_all[idx] for idx, x in enumerate(on_residential)
        ]
        dist_cs["residential"] = (
                total_dist_cs["on_residential"] / on_all_cs
        )

        # On bike paths
        on_bike = total_dist["on_bike_path"]
        dist["bike paths"] = [x / on_all[idx] for idx, x in enumerate(on_bike)]
        dist_cs["bike paths"] = (
                total_dist_cs["on_bike_path"] / on_all_cs
        )

    if not buildup:
        for st, len_on_st in dist.items():
            dist[st] = list(reversed(len_on_st))
    return dist, dist_cs


def sum_total_cost(cost: list, buildup: bool = False) -> list:
    """Sums up all total cost up to each step.

    Parameters
    ----------
    cost : list
        Costs per step
    buildup : bool (Default value = False)
        Reversed algorithm used True/False

    Returns
    -------
    list
        Summed cost

    """
    if not buildup:
        cost = list(reversed(cost))
    total_cost = [sum(cost[:i]) for i in range(0, len(cost))]
    return total_cost


def get_end(
        tdt: dict,
        tdt_opt: dict,
        tdt_base: dict,
        buildup: bool = False,
        felt: bool = True
    ) -> int:
    """Returns the index where the bikeability reaches 1.

    Parameters
    ----------
    tdt : dict
        total distance traveled
    tdt_opt : dict
        total distance traveled for optimal state
    tdt_base : dict
        total distance traveled for base state
    buildup : bool (Default value = False)
        Reversed algorithm used True/False
    felt : bool
        If felt distance (Default value = True)

    Returns
    -------
    int
        Index where bikeability reaches 1

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


def get_street_type_ratio(G: nx.MultiGraph | nx.MultiDiGraph) -> dict:
    """Gets the ratios for the different street types in the given graph.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network.

    Returns
    -------
    output : dict
        Street type ratio in dict keyed by street type

    """
    G = G.to_undirected()
    st_len = {"primary": 0, "secondary": 0, "tertiary": 0, "residential": 0}
    total_len = 0
    for edge in G.edges:
        e_st = get_street_type(G, edge)
        e_len = G[edge[0]][edge[1]][0]["length"]
        st_len[e_st] += e_len
        total_len += e_len
    st_len_norm = {k: v / total_len for k, v in st_len.items()}
    return st_len_norm


def calc_polygon_area(
        polygon: shapely.Polygon,
        remove: Iterable[shapely.Polygon] | None = None,
        unit: str ="sqkm"
    ) -> float:
    """Calculates the area of a given lat/long Polygon.

    Parameters
    ----------
    polygon : shapely.Polygon
        Polygon to calculate the area of
    remove : list[shapely.Polygon]
        Polygons inside the original polygon to exclude from the
        area calculation (Default value = None)
    unit : str
        Unit in which the area is returned km^2 = 'sqkm' or m^2 = 'sqm' (Default value = "sqkm")

    Returns
    -------
    output : float

    """
    if (not isinstance(remove, list)) ^ (remove is None):
        remove = [remove]
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init="EPSG:4326"),
            pyproj.Proj(
                proj="aea", lat_1=polygon.bounds[1], lat_2=polygon.bounds[3]
            ),
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
                    pyproj.Proj(
                        proj="aea", lat_1=p.bounds[1], lat_2=p.bounds[3]
                    ),
                ),
                p,
            )
            remove_area += a_r.area

    if unit == "sqkm":
        return (geom_area.area - remove_area) / 1000000
    if unit == "sqm":
        return geom_area.area - remove_area


def get_edge_color_bp(
        G: nx.MultiGraph | nx.MultiDiGraph,
        edges: list,
        color: str,
        ex_inf: bool,
        color_ex_inf: str,
        color_unused: str
    ) -> list:
    """

    Parameters
    ----------
    G : nx.MultiGraph or nx.MultiDiGraph
        Street network
    edges : list
        Edges to color with 'color'
    color : str (hex code of color)
        Edge color
    ex_inf : bool
        If existing infrastructure should be colored separately
    color_ex_inf : str (hex code of color)
        Color for existing infrastructure
    color_unused : str (hex code of color)
        Color for unused edges

    Returns
    -------
    output : list
    
    """
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


def get_edge_color(
        G: nx.MultiGraph | nx.MultiDiGraph,
        edges: list,
        attr: int | float | str,
        color: str
    ) -> list:
    """Return edge color list for G, edges have the given color if they are part of the edges list and therefore have the given attribute, otherwise they have the color '#999999'.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    edges : list
        Edges which have the attribute
    attr : int, float or str
        Attribute for the coloring (e.g. bike path)
    color : color (e.g. hexcode)
        Color if edge has attribute

    Returns
    -------
    list
        Edge colors for graph G.

    """
    nx.set_edge_attributes(G, False, attr)
    for edge in edges:
        G[edge[0]][edge[1]][0][attr] = True
        G[edge[1]][edge[0]][0][attr] = True
    return [
        color if data[attr] else "#999999"
        for u, v, data in G.edges(keys=False, data=True)
    ]


def get_edge_color_st(G: nx.MultiGraph | nx.MultiDiGraph, colors: dict) -> list:
    """Return edge color list, to color the edges depending on their street type.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    colors : dict
        Street type colors

    Returns
    -------
    list
        Edge colors for graph G

    """
    return [
        colors[get_street_type(G, e)] for e in G.edges()
    ]


def plot_barh(
        data: dict,
        colors: dict,
        save: str,
        x_label: str = "",
        title: str = "",
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 150,
    ):
    """Plot and save a horizontal bar.

    Parameters
    ----------
    data : dict
        Data to plot as dictionary.
    colors : dict
        Colors for the data
    save : str
        Save location
    figsize : tuple[float, float]
        Size of figure (width, height) (Default value = (10, 10))
    x_label : str
        Label for the x-axis (Default value = "")
    title : str
        Title of the plot (Default value = "")
    dpi : int
        dpi of the plot (Default value = 150)

    Returns
    -------

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
                "white"
                if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25
                else "black"
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
        data: dict,
        stacks: list,
        colors: list,
        save: str,
        title: str = "",
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 150,
        legend=False,
    ):
    """Plot and save a stacked horizontal bar.

    Parameters
    ----------
    data : dict
        Data to plot as dictionary.
    stacks :
        Stacks to plot.
    colors : list
        Colors for the data
    save : str
        Save location
    title : str
        Title of the plot (Default value = "")
    figsize : tuple[float, float]
        Size of figure (width, height) (Default value = (10, 10))
    dpi : int
        dpi of the plot (Default value = 150)
    legend : bool
        If legend should be plotted or not (Default value = False)

    Returns
    -------
    type
        None

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
        ax.barh(
            labels, widths, left=starts, height=0.5, label=colname, color=color
        )
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = (
            "white" if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25 else "black"
        )
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
        data: dict,
        colors: dict,
        save: str,
        y_label: str = "",
        title: str = "",
        ymin: float = -0.1,
        ymax: float = 0.7,
        xticks: bool = True,
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 150,
    ):
    """Plot and save a vertical bar.

    Parameters
    ----------
    data : dict
        Data to plot as dictionary.
    colors : dict
        Colors for the data
    save : str
        Save location
    figsize : tuple
        Size of figure (width, height) (Default value = None)
    y_label : str
        Label for the x-axis (Default value = "")
    title : str
        Title of the plot (Default value = "")
    ymin : float
        Minimal y value for axis (Default value = -0.1)
    ymax : float
        Maximal y value for axis (Default value = 0.7)
    xticks : bool
        Plot x ticks or not (Default value = True)
    dpi : int
        dpi of the plot (Default value = 150)

    Returns
    -------

    """
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
        labels: list,
        data: dict,
        colors: dict,
        title: str = "",
        ylabel: str = "",
        save: str = "",
        width: float = 0.8,
        figsize: tuple[float, float] = (10, 12),
        dpi=150,
    ):
    """Plot and save a stacked vertical bar.

    Parameters
    ----------
    labels : list
        Labels for the bars
    data : dict
        Data to plot as dictionary.
    colors : dict
        Colors for the data
    title : str
        Title of the plot (Default value = "")
    ylabel : str
        Label for y-axis (Default value = "")
    save : str
        Save location (Default value = "")
    width : float
        Width of bars (Default value = 0.8)
    figsize : tuple
        Size of figure (width, height) (Default value = (10, 12))
    dpi : int
        dpi of the plot (Default value = 150)

    Returns
    -------

    """
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
                    "white"
                    if (r * 0.299 + g * 0.587 + b * 0.114) < 0.25
                    else "black"
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
        save_path: str,
        bins = None,
        cumulative: bool = False,
        density: bool = False,
        xlabel="",
        ylabel="",
        xlim=None,
        xaxis: bool = True,
        xticks: dict | None = None,
        cm: ListedColormap | None = None,
        figsize=(4, 4),
        dpi=150,
    ):
    """Plot and save a histogram.

    Parameters
    ----------
    data :
        Data to plot.
    save_path: str
        Path to save the figure.
    bins :
        Bins for the histogram. (see matplotlib.pyplot.hist)
    cumulative : bool
        To plot a cumulative histogram or not. (see matplotlib.pyplot.hist)
    density : bool
        To draw and return density or not. (see matplotlib.pyplot.hist)
    xlabel : str
        Label for x-axis (Default value = "")
    ylabel : str
        Label for y-axis (Default value = "")
    xlim : tuple
        Limits for x-axis (Default value = None)
    xaxis : bool
        Plot x-axis or not. (Default value = False)
    xticks : dict
        Ticks and labels for x-axis. (Default value = None)
    cm : ListedColormap or None
        Coloring for the bars (Default value = None)
    figsize : tuple
        Size of figure (width, height) (Default value = None)
    dpi : int
        Dpi of the plot (Default value = 150)

    Returns
    -------

    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        max_d = max(data)
        min_d = min(data)
        r = magnitude(max_d)
        ax.set_xlim(left=0.0, right=round(max_d + 0.1 * max_d, -(r - 1)))
    if bins is None:
        max_d = max(data)
        min_d = min(data)
        r = magnitude(max_d)
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
        if isinstance(cm, dict):
            for bar in ax.containers[0]:
                x = bar.get_x() + 0.5 * bar.get_width()
                bar.set_color(cm[min(i for i in cm.keys() if i > x)])
        else:
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


def plot_graph(
        G: nx.MultiGraph | nx.MultiDiGraph,
        *,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8, 8),
        bgcolor: str = "#111111",
        node_color: str | Sequence[str] = "w",
        node_size: float | Sequence[float] = 15,
        node_alpha: float | None = None,
        node_edgecolor: str | Iterable[str] = "none",
        node_zorder: int = 1,
        edge_color: str | Iterable[str] = "#999999",
        edge_linewidth: float | Sequence[float] = 1,
        edge_alpha: float | None = None,
        edge_zorder: int | Iterable[int] = 1,
        bbox: tuple[float, float, float, float] | None = None,
        show: bool = True,
        close: bool = False,
        save: bool = False,
        filepath: str | Path | None = None,
        dpi: int = 300,
    ) -> tuple[Figure, Axes]:
    """Adaption of the plot_graph function of the osmnx package in order to have a
    different zorders for different edges. Visualize a graph.

    Parameters
    ----------
    G
        Input graph.
    ax
        If not None, plot on this pre-existing axes instance.
    figsize
        If `ax` is None, create new figure with size `(width, height)`.
    bgcolor
        Background color of the figure.
    node_color
        Color(s) of the nodes.
    node_size
        Size(s) of the nodes. If 0, then skip plotting the nodes.
    node_alpha
        Opacity of the nodes. If you passed RGBa values to `node_color`, set
        `node_alpha=None` to use the alpha channel in `node_color`.
    node_edgecolor
        Color(s) of the nodes' markers' borders.
    node_zorder
        The zorder to plot nodes. Edges are always 1, so set `node_zorder=0`
        to plot nodes beneath edges.
    edge_color
        Color(s) of the edges' lines.
    edge_linewidth
        Width(s) of the edges' lines. If 0, then skip plotting the edges.
    edge_alpha
        Opacity of the edges. If you passed RGBa values to `edge_color`, set
        `edge_alpha=None` to use the alpha channel in `edge_color`.
    edge_zorder
        The zorder to plot edges. If single integer all edges have the same zorder.
    bbox
        Bounding box as `(left, bottom, right, top)`. If None, calculate it
        from spatial extents of plotted geometries.
    show
        If True, call `pyplot.show()` to show the figure.
    close
        If True, call `pyplot.close()` to close the figure.
    save
        If True, save the figure to disk at `filepath`.
    filepath
        The path to the file if `save` is True. File format is determined from
        the extension. If None, save at `settings.imgs_folder/image.png`.
    dpi
        The resolution of saved file if `save` is True.

    Returns
    -------
    fig, ax

    """

    max_node_size = max(node_size) if hasattr(node_size, "__iter__") else node_size
    max_edge_lw = max(edge_linewidth) if hasattr(edge_linewidth, "__iter__") else edge_linewidth
    if max_node_size <= 0 and max_edge_lw <= 0:  # pragma: no cover
        msg = "Either node_size or edge_linewidth must be > 0 to plot something."
        raise ValueError(msg)

    # create fig, ax as needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor, frameon=False)
        ax.set_facecolor(bgcolor)
    else:
        fig = ax.figure

    if max_edge_lw > 0:
        # plot the edges' geometries
        gdf_edges = ox.convert.graph_to_gdfs(G, nodes=False)
        gdf_edges["zorder"] = edge_zorder
        gdf_edges["color"] = edge_color
        gdf_edges["linewidth"] = edge_linewidth
        gdf_edges["alpha"] = edge_alpha
        gdf_edges.sort_values("zorder", ascending=False, inplace=True)

        color = gdf_edges.color if edge_color is not None else None
        linewidth = gdf_edges.linewidth if edge_linewidth is not None else None
        alpha = gdf_edges.alpha if edge_alpha is not None else None

        ax = gdf_edges.plot(ax=ax, color=color, lw=linewidth, alpha=alpha, zorder=1, facecolor=bgcolor)

    if max_node_size > 0:
        # scatter plot the nodes' x/y coordinates
        gdf_nodes = ox.convert.graph_to_gdfs(G, edges=False, node_geometry=False)[["x", "y"]]
        ax.scatter(
            x=gdf_nodes["x"],
            y=gdf_nodes["y"],
            s=node_size,
            c=node_color,
            alpha=node_alpha,
            edgecolor=node_edgecolor,
            zorder=node_zorder,
        )

    # get spatial extents from bbox parameter or the edges' geometries
    padding = 0.0
    if bbox is None:
        try:
            left, bottom, right, top = gdf_edges.total_bounds
        except NameError:
            left, bottom = gdf_nodes.min()
            right, top = gdf_nodes.max()
        bbox = left, bottom, right, top
        padding = 0.02  # pad 2% to not cut off peripheral nodes' circles

    # configure axis appearance, save/show figure as specified, and return
    ax = _config_ax(ax, G.graph["crs"], bbox, padding)  # type: ignore[arg-type]
    fig, ax = _save_and_show(
        fig=fig,
        ax=ax,
        show=show,
        close=close,
        save=save,
        filepath=filepath,
        dpi=dpi,
    )
    return fig, ax
