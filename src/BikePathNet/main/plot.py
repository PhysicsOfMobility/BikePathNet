"""
This module includes all necessary functions for the plotting functionality.
"""
import json
import h5py
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from scipy import interpolate
from ..helper.data_helper import get_polygon_from_json, average_hom_demand
from ..helper.algorithm_helper import calc_single_state
from ..helper.plot_helper import *
from ..helper.setup_helper import create_default_params, create_default_paths


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
    if params is None:
        params = create_default_params()

    fig, ax = plt.subplots(figsize=params["figs_snetwork"],
                           dpi=params["dpi"])
    ox.plot_graph(G, ax=ax, bgcolor='#ffffff', show=False, close=False,
                  node_color=params["nc_snetwork"],
                  node_size=params["ns_snetwork"], node_zorder=3,
                  edge_color=params["ec_snetwork"],
                  edge_linewidth=params["ew_snetwork"])
    if params["titles"]:
        fig.suptitle(f'Graph used for {city.capitalize()}',
                     fontsize=params["fs_title"])

    plt.savefig(f'{plot_folder}{save}_street_network.{params["plot_format"]}')


def plot_used_nodes(city, save, G, trip_nbrs, stations, plot_folder,
                    params=None):
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
    if params is None:
        params = create_default_params()

    nodes = {n: 0 for n in G.nodes()}
    for s_node in G.nodes():
        for e_node in G.nodes():
            if e_node == s_node:
                continue
            if (s_node, e_node) in trip_nbrs:
                nodes[s_node] += trip_nbrs[(s_node, e_node)]
                nodes[e_node] += trip_nbrs[(s_node, e_node)]

    nodes = {n: int(t / params["stat_usage_norm"]) for n, t in nodes.items()}

    trip_count = sum(trip_nbrs.values())
    station_count = len(stations)

    max_n = max(nodes.values())

    n_rel = {key: value for key, value in nodes.items()}
    ns = [params["ns_station_usage"] if n in stations else 0
          for n in G.nodes()]

    for n in G.nodes():
        if n not in stations:
            n_rel[n] = max_n + 1
    min_n = min(n_rel.values())

    r = magnitude(max_n)

    hist_data = [value for key, value in n_rel.items() if value != max_n + 1]
    hist_save = f'{plot_folder}{save}_stations_usage_distribution'
    hist_xlim = (0.0, round(max_n, -(r - 1)))

    cmap = plt.cm.get_cmap(params["cmap_nodes"])

    plot_histogram(hist_data, hist_save,
                   xlabel='total number of trips per year',
                   ylabel='number of stations', xlim=hist_xlim, bins=25,
                   cm=cmap, params=params)

    cmap = ['#808080'] + [rgb2hex(cmap(n)) for n in
                          reversed(np.linspace(1, 0, max_n, endpoint=True))] \
           + ['#ffffff']
    color_n = [cmap[v] for k, v in n_rel.items()]

    fig2, ax2 = plt.subplots(dpi=params["dpi"],
                             figsize=params["figs_station_usage"])
    ox.plot_graph(G, ax=ax2, bgcolor='#ffffff',
                  edge_linewidth=params["edge_lw"],
                  node_color=color_n, node_size=ns, node_zorder=3,
                  show=False, close=False,
                  edge_color=params["ec_station_usage"])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(params["cmap_nodes"]),
                               norm=plt.Normalize(vmin=0, vmax=max_n))

    cbaxes = fig2.add_axes([0.1, 0.05, 0.8, 0.03])
    # divider = make_axes_locatable(ax2)
    # cbaxes = divider.append_axes('bottom', size="5%", pad=0.05)

    if min_n <= 0.1 * max_n:
        r = magnitude(max_n)
        cbar = fig2.colorbar(sm, orientation='horizontal', cax=cbaxes,
                             ticks=[0, round(max_n / 2), max_n])
        cbar.ax.set_xticklabels([0, int(round(max_n / 2, -(r - 2))),
                                 round(max_n, -(r - 1))])
    else:
        max_r = magnitude(max_n)
        min_r = magnitude(min_n)
        cbar = fig2.colorbar(sm, orientation='horizontal', cax=cbaxes,
                             ticks=[0, min_n, round(max_n / 2), max_n])
        cbar.ax.set_xticklabels([0, round(min_n, -(min_r - 2)),
                                 int(round(max_n / 2, -(max_r - 2))),
                                 round(max_n, -(max_r - 2))])

    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    cbar.ax.set_xlabel('total number of trips per year',
                       fontsize=params["fs_axl"])

    if params["titles"]:
        fig2.suptitle(f'{city.capitalize()}, Stations: {station_count}, '
                      f'Trips: {trip_count}', fontsize=params["fs_title"])

    fig2.savefig(f'{plot_folder}{save}_stations_used.{params["plot_format"]}',
                 bbox_inches='tight')

    plt.close('all')


def plot_bp_comparison(city, save, G, ee_algo, ee_cs, bpp_algo, bpp_cs,
                       stations, plot_folder, mode='diff', params=None):
    """
    Plots the comparison of bike path networks for the algorithm and the p+s
    state.
    :param city: Name of the city
    :type city: str
    :param save: save name
    :type save: str
    :param G: graph
    :type G: osmx graph
    :param ee_algo: Edited edges by the algorithm
    :type ee_algo: list
    :param ee_cs: Edited edges for p+s
    :type ee_cs: list
    :param bpp_algo: Bike path percentage for the algorithm
    :type bpp_algo: list
    :param bpp_cs: Bike path percentage for p+s
    :type bpp_cs: float
    :param stations: Stations used
    :type stations: list
    :param plot_folder: Folder to save plots to.
    :type plot_folder: str
    :param mode: Mode to plot the comparison. Only algorithm: 'algo',
    only p+s: 'p+s', difference: 'diff.
    :type mode: str
    :param params: Parameters for plotting.
    :type params: dict
    :return: None
    """
    nx.set_edge_attributes(G, False, 'algo')
    nx.set_edge_attributes(G, False, 'cs')

    ns = [params["ns_bp_comp"] if n in stations else 0 for n in G.nodes()]

    ee_algo = list(reversed(ee_algo))
    bpp_algo = list(reversed(bpp_algo))

    idx = min(range(len(bpp_algo)), key=lambda i: abs(bpp_algo[i] - bpp_cs))

    ee_algo_cut = ee_algo[:idx]
    for edge in ee_algo_cut:
        G[edge[0]][edge[1]][0]['algo'] = True
        G[edge[1]][edge[0]][0]['algo'] = True
    for edge in ee_cs:
        G[edge[0]][edge[1]][0]['cs'] = True
        G[edge[1]][edge[0]][0]['cs'] = True

    ec = []
    unused = []
    ee_algo_only = []
    ee_cs_only = []
    ee_both = []

    len_algo = 0
    len_cs = 0
    len_both = 0

    if mode == 'algo':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['algo']:
                ec.append(params["color_algo"])
            else:
                ec.append(params["color_unused"])
                unused.append((u, v, k))
    elif mode == 'p+s':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['cs']:
                ec.append(params["color_cs"])
            else:
                ec.append(params["color_unused"])
                unused.append((u, v, k))
    elif mode == 'diff':
        for u, v, k, data in G.edges(keys=True, data=True):
            if data['algo'] and data['cs']:
                ec.append(params["color_both"])
                ee_both.append((u, v, k))
                len_both += data['length']
            elif data['algo'] and not data['cs']:
                ec.append(params["color_algo"])
                ee_algo_only.append((u, v, k))
                len_algo += data['length']
            elif not data['algo'] and data['cs']:
                ec.append(params["color_cs"])
                ee_cs_only.append((u, v, k))
                len_cs += data['length']
            else:
                ec.append(params["color_unused"])
                unused.append((u, v, k))
    else:
        print('You have to choose between algo, p+s and diff.')

    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_bp_comp"])
    ox.plot_graph(G, bgcolor='#ffffff', ax=ax,
                  node_size=ns, node_color=params["nc_pb_evo"],
                  node_zorder=3, edge_linewidth=0.6, edge_color=ec,
                  show=False, close=False)
    if params["legends"]:
        lw_leg = params["lw_legend_bp_evo"]
        if mode == 'algo':
            leg = [Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_unused"], lw=lw_leg)]
            ax.legend(leg, ['Algorithm', 'None'],
                      bbox_to_anchor=(0, -0.05, 1, 1), loc=3, ncol=2,
                      mode="expand", borderaxespad=0.,
                      fontsize=params["fs_legend"])
        elif mode == 'p+s':
            leg = [Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_unused"], lw=lw_leg)]
            ax.legend(leg, ['Primary + Secondary', 'None'],
                      bbox_to_anchor=(0, -0.05, 1, 1), loc=3, ncol=2,
                      mode="expand", borderaxespad=0.,
                      fontsize=params["fs_legend"])
        elif mode == 'diff':
            leg = [Line2D([0], [0], color=params["color_both"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_algo"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_cs"], lw=lw_leg),
                   Line2D([0], [0], color=params["color_unused"], lw=lw_leg)]
            ax.legend(leg, ['Both', 'Algorithm', 'Primary+Secondary', 'None'],
                      bbox_to_anchor=(0, -0.05, 1, 1), loc=3, ncol=4,
                      mode="expand", borderaxespad=0.,
                      fontsize=params["fs_legend"])
    if params["titles"]:
        if mode == 'algo':
            ax.set_title(f'{city}: Algorithm', fontsize=params["fs_title"])
        elif mode == 'p+s':
            ax.set_title(f'{city}: Primary/Secondary',
                         fontsize=params["fs_title"])
        elif mode == 'diff':
            ax.set_title(f'{city}: Comparison', fontsize=params["fs_title"])

    plt.savefig(f'{plot_folder}{save}_bp_build_{mode}.{params["plot_format"]}',
                bbox_inches='tight')
    plt.close(fig)


def plot_mode(city, save, data, data_ps, nxG_plot, stations, end, hf_group,
              plot_folder, params=None):
    """
    Plots the results for one mode for the given city. If no 'params' are given
    the default ones are created and used.

    :param city: Name of the city
    :type city: str
    :param save: Save name of the city.
    :type save: str
    :param data: hdf group with the algorithm data
    :type data: hdf group
    :param data_ps: data for P+S state
    :type data_ps: list
    :param nxG_plot: Graph to plot in
    :type nxG_plot: networkx (Multi)(Di)graph
    :param stations: List of stations
    :type stations: list
    :param end: Point to cut at
    :type end: int
    :param hf_group: hdf group for saving the evaluated data
    :type hf_group: hdf group
    :param plot_folder: Path to folder for storing plots
    :type plot_folder: str
    :param params: Dict with the params for plots etc., check 'params.py' in
    the example folder.
    :type params: dict
    :return: None
    """
    if params is None:
        params = create_default_params()

    bike_paths_ps = data_ps[0]
    cost_ps = data_ps[1]
    bike_path_perc_ps = data_ps[2]
    trdt_ps = data_ps[3]
    tfdt_ps = data_ps[4]

    edited_edges_nx = [(i[0], i[1]) for i in data['ee_nx'][()]]
    bike_path_perc = data['bpp'][()]
    cost = data['cost'][()]
    total_real_distance_traveled = json.loads(data['trdt'][()])
    total_felt_distance_traveled = json.loads(data['tfdt'][()])

    trdt, trdt_ps = total_distance_traveled_list(total_real_distance_traveled,
                                                  trdt_ps)
    tfdt, tfdt_ps = total_distance_traveled_list(total_felt_distance_traveled,
                                                  tfdt_ps)

    bpp = list(reversed(bike_path_perc))

    trdt_min = min(trdt['all'])
    trdt_max = max(trdt['all'])
    tfdt_min = min(tfdt['all'])
    tfdt_max = max(tfdt['all'])
    ba = [1 - (i - tfdt_min) / (tfdt_max - tfdt_min) for i in tfdt['all']]

    ba_ps = 1 - (tfdt_ps['all'] - tfdt_min) / (tfdt_max - tfdt_min)

    trdt_st = {st: len_on_st for st, len_on_st in trdt.items()
               if st not in ['street', 'all']}
    trdt_st_ps = {st: len_on_st for st, len_on_st in trdt_ps.items()
                   if st not in ['street', 'all']}

    bpp_cut = [i / bpp[end] for i in bpp[:end+1]]
    bpp_ps = bike_path_perc_ps / bpp[end]

    bpp_x = min(bpp_cut, key=lambda x: abs(x - bpp_ps))
    bpp_idx = next(x for x, val in enumerate(bpp_cut) if val == bpp_x)
    ba_y = ba[bpp_idx]

    bpp_10 = min(bpp_cut, key=lambda x: abs(x - 0.10))
    bpp_10_idx = next(x for x, val in enumerate(bpp_cut) if val == bpp_10)
    ba_10 = ba[bpp_10_idx]

    cost_y = min(cost[:end+1], key=lambda x: abs(x - cost_ps))
    cost_idx = next(x for x, val in enumerate(cost[:end+1]) if val == cost_y)
    cost_x = bpp_cut[cost_idx]

    cut = next(x for x, val in enumerate(ba) if val >= 1)
    total_cost, cost_ps = sum_total_cost(cost, cost_ps)
    cost_ps = cost_ps / total_cost[end]

    max_bpp = max(bpp[end], bpp[cut])

    # Plotting
    fig1, ax1 = plt.subplots(dpi=params["dpi"], figsize=params["figs_ba_cost"])
    ax12 = ax1.twinx()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax12.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax12.set_ylim(0.0, 1.0)

    ax1.plot(bpp_cut, ba[:end+1], c=params["c_ba"], label='bikeability',
             lw=params["lw_ba"])
    ax1.plot(bpp_ps, ba_ps, c=params["c_ba"], ms=params["ms_ba"],
             marker=params["m_ba"])
    xmax, ymax = coord_transf(bpp_ps, max([ba_y, ba_ps]),
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_ps, ymax=ymax, ymin=0, c=params["c_ba"], ls='--',
                alpha=0.5, lw=params["lw_ba"])
    ax1.axhline(y=ba_ps, xmax=xmax, xmin=0, c=params["c_ba"], ls='--',
                alpha=0.5, lw=params["lw_ba"])
    ax1.axhline(y=ba_y, xmax=xmax, xmin=0, c=params["c_ba"], ls='--',
                alpha=0.5, lw=params["lw_ba"])

    xmax_10, ymax_10 = coord_transf(bpp_10, ba_10,
                                    xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_10, ymax=ymax_10, ymin=0, c=params["c_ba"], ls='--',
                alpha=0.5, lw=params["lw_ba"])
    ax1.axhline(y=ba_10, xmax=xmax_10, xmin=0, c=params["c_ba"], ls='--',
                alpha=0.5, lw=params["lw_ba"])

    ax1.set_ylabel(r'bikeability $b(\lambda)$', fontsize=params["fs_axl"],
                   color=params["c_ba"])
    ax1.tick_params(axis='y', labelsize=params["fs_ticks"],
                    labelcolor=params["c_ba"])
    ax1.tick_params(axis='x', labelsize=params["fs_ticks"])
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax12.plot(bpp_cut, [x / total_cost[end] for x in total_cost[:end+1]],
              c=params["c_cost"], label='total cost', lw=params["lw_cost"])
    ax12.plot(bpp_ps, cost_ps, c=params["c_cost"], ms=params["ms_cost"],
              marker=params["m_cost"])
    xmin, ymax = coord_transf(bpp_ps, cost_ps,
                              xmax=1, xmin=0, ymax=1, ymin=0)
    ax1.axvline(x=bpp_ps, ymax=ymax, ymin=0, c=params["c_cost"], ls='--',
                alpha=0.5, lw=params["lw_cost"])
    ax1.axhline(y=cost_ps, xmax=1, xmin=xmin, c=params["c_cost"], ls='--',
                alpha=0.5, lw=params["lw_cost"])
    ax1.axhline(y=cost_y, xmax=1, xmin=xmin, c=params["c_cost"], ls='--',
                alpha=0.5, lw=params["lw_cost"])

    ax12.set_ylabel('normalized cost', fontsize=params["fs_axl"],
                    color=params["c_cost"])
    ax12.tick_params(axis='y', labelsize=params["fs_ticks"],
                     labelcolor=params["c_cost"])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax12.yaxis.set_minor_locator(AutoMinorLocator())
    ax12.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax1.set_xlabel(r'normalized relative length of bike paths $\lambda$',
                   fontsize=params["fs_axl"])
    if params["titles"]:
        ax1.set_title('Bikeability and Cost', fontsize=params["fs_title"])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(False)
    ax12.grid(False)

    handles = ax1.get_legend_handles_labels()[0]
    handles.append(ax12.get_legend_handles_labels()[0][0])
    if params["legends"]:
        ax1.legend(handles=handles, loc='lower right',
                   fontsize=params["fs_legend"])

    fig1.savefig(f'{plot_folder}{save}_ba_tc.{params["plot_format"]}',
                 bbox_inches='tight')

    comp_st_driven = {st: [len_on_st[bpp_idx], trdt_st_ps[st]]
                      for st, len_on_st in trdt_st.items()}
    plot_barv_stacked(['Algorithm', 'P+S'], comp_st_driven, params["c_st"],
                      width=0.3, title='',
                      save=f'{plot_folder}{save}_comp_st_driven',
                      params=params)

    for bp_mode in ['algo', 'p+s', 'diff']:
        plot_bp_comparison(city=city, save=save, G=nxG_plot,
                           ee_algo=edited_edges_nx, ee_cs=bike_paths_ps,
                           bpp_algo=bike_path_perc, bpp_cs=bike_path_perc_ps,
                           stations=stations, plot_folder=plot_folder,
                           mode=bp_mode, params=params)

    plt.close('all')

    print(f'{save}: b={ba_y:4.2f}, b_p+s={ba_ps:4.2f}')

    hf_group['edited edges'] = edited_edges_nx
    hf_group['end'] = end
    hf_group['bpp'] = bpp_cut
    hf_group['bpp at end'] = bpp[end]
    hf_group['bpp complete'] = bike_path_perc
    hf_group['ba'] = ba[:end+1]
    hf_group['ba complete'] = ba
    hf_group['ba for comp'] = ba_y
    hf_group['cost'] = total_cost[:end+1]
    hf_group['tfdt'] = tfdt['all']
    hf_group['tfdt max'] = tfdt_max
    hf_group['tfdt min'] = tfdt_min
    hf_group['trdt max'] = trdt_max
    hf_group['trdt min'] = trdt_min
    hf_group['los'] = trdt['street'][:end+1]

    return bpp_ps, ba_ps, cost_ps


def plot_city(city, save, paths=None, params=None):
    """
    Plots the results for the given city. If no 'paths' or 'params' are given
    the default ones are created and used.

    :param city: Name of the city
    :type city: str
    :param save: Save name of the city.
    :type save: str
    :param paths: Dict with the paths for data, plots etc., check 'paths.py'
    in the example folder.
    :type paths: dict
    :param params: Dict with the params for plots etc., check 'params.py' in
    the example folder.
    :type params: dict
    :return: None
    """
    if paths is None:
        paths = create_default_paths()

    if params is None:
        params = create_default_params()

    # Define city specific folders
    comp_folder = f'{paths["comp_folder"]}/'
    plot_folder = f'{paths["plot_folder"]}results/{save}/'
    input_folder = f'{paths["input_folder"]}{save}/'
    output_folder = f'{paths["output_folder"]}{save}/'

    # Create non existing folders
    Path(comp_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    hf_comp = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'w')
    hf_comp.attrs['city'] = city

    plt.rcdefaults()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]})


    hf_demand = h5py.File(f'{input_folder}{save}_demand.hdf5', 'r')
    trip_nbrs = {(int(k1), int(k2)): v[()] for k1 in list(hf_demand.keys())
                 for k2, v in hf_demand[k1].items()}
    hf_demand.close()

    trips = sum(trip_nbrs.values())
    hf_comp.attrs['total trips'] = trips
    utrips = len(trip_nbrs.keys())
    hf_comp.attrs['unique trips'] = utrips

    stations = [station for trip_id, nbr_of_trips in trip_nbrs.items() for
                station in trip_id]
    stations = list(set(stations))
    hf_comp.attrs['nbr of stations'] = len(stations)
    hf_comp['stations'] = stations

    nxG = ox.load_graphml(filepath=f'{input_folder}{save}.graphml')
    hf_comp.attrs['nodes'] = len(nxG.nodes)
    hf_comp.attrs['edges'] = len(nxG.edges)
    nxG_plot = nxG.to_undirected()
    nxG_calc = nx.Graph(nxG.to_undirected())

    polygon_path = f'{paths["polygon_folder"]}{save}.json'
    polygon = get_polygon_from_json(polygon_path)

    area = calc_polygon_area(polygon)
    hf_comp.attrs['area'] = area

    print(f'{save}: area={area:.0f}km^2, stations={len(stations)}, '
          f'trips={trips}')

    plot_used_nodes(city=city, save=save, G=nxG_plot, trip_nbrs=trip_nbrs,
                    stations=stations, plot_folder=plot_folder, params=params)

    ps_bike_paths = [e for e in nxG_calc.edges() if
                     get_street_type_cleaned(nxG_calc, e, multi=False)
                     in ['primary', 'secondary']]
    data_ps = calc_single_state(nxG=nxG_calc, trip_nbrs=trip_nbrs,
                                 bike_paths=ps_bike_paths, params=params)

    data = h5py.File(f'{output_folder}{save}_data.hdf5', 'r')['all']

    if params["cut"]:
        end = get_end(json.loads(data['trdt'][()]), data_ps[3])
    else:
        end = len(data['bpp'][()]) - 1

    grp_algo = hf_comp.create_group('algorithm')
    bpp_ps, ba_ps, cost_ps, = \
        plot_mode(city=city, save=save, data=data, data_ps=data_ps,
                  nxG_plot=nxG_plot, stations=stations, end=end,
                  hf_group=grp_algo, plot_folder=plot_folder,
                  params=params)

    grp_ps = hf_comp.create_group('p+s')
    grp_ps['bpp'] = bpp_ps
    grp_ps['ba'] = ba_ps
    grp_ps['cost'] = cost_ps


def plot_comp_hom_demand(city, save, bpp_ed, bpp_hom, ba_ed, ba_hom,
                         params, paths):
    """
    Plotting the comparison between empirical demand and homogenized demand.
    :param city: Name of the city
    :type city: str
    :param save: Save name of the city
    :type save: str
    :param bpp_ed: Bike path percentage for emp. demand.
    :type bpp_ed: list
    :param bpp_hom: Bike path percentage for hom. demand.
    :type bpp_hom: list
    :param ba_ed: Bikeability for emp. demand.
    :type ba_ed: list
    :param ba_hom: Bikeability for hom. demand.
    :type ba_hom: list
    :param params: Parameters for plotting
    :type params: dict
    :param paths: Paths to load and save from/to.
    :type paths: dict
    :return: None
    """
    fig, ax = plt.subplots(dpi=params["dpi"], figsize=params["figs_hom_comp"])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.plot(bpp_ed, ba_ed, c=params["c_ed"], lw=params["lw_ed"],
            label='emp. demand')
    ax.plot(bpp_hom, ba_hom, c=params["c_hom"], lw=params["lw_hom"],
            label='hom. demand')
    poly = ax.fill(np.append(bpp_ed, bpp_hom[::-1]),
                   np.append(ba_ed, ba_hom[::-1]),
                   color=params["c_hom_ed_area"],
                   alpha=params["a_hom_ed_area"])
    print(f'{save}: beta_hom={Polygon(poly[0].get_xy()).area:.3f}')
    ax.set_ylabel(r'bikeability $b(\lambda)$', fontsize=params["fs_axl"])
    ax.tick_params(axis='y', labelsize=params["fs_ticks"], width=0.5)
    ax.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel(r'normalized relative length of bike paths $\lambda$',
                  fontsize=params["fs_axl"])
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    if params["titles"]:
        ax.set_title(f'{city}', fontsize=params["fs_title"])
    if params["legends"]:
        ax.legend(loc='lower right', fontsize=params["fs_legend"])

    fig.savefig(f'{paths["plot_folder"]}results/{save}/'
                f'{save}_ba_hom_comp.{params["plot_format"]}',
                bbox_inches='tight')


def plot_city_hom_demand(city, save, paths, params, hom_pattern='hom',
                         nbr_of_hom_sets=10):
    """
    Average the results of the hom demand and plot the comparison with the
    emp. demand
    :param city: Name of the city
    :type city: str
    :param save: save name of the city
    :type save: str
    :param paths: Paths for loading and saving from/to.
    :type paths: dict
    :param params: Params for plotting
    :type params: dict
    :param hom_pattern: naming pattern of the hom. sets.
    :type hom_pattern: str
    :param nbr_of_hom_sets: Number of hom. demand sets.
    :type nbr_of_hom_sets: int
    :return:
    """
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]})

    comp_folder = f'{paths["comp_folder"]}'

    # Plot the comparison between the empirical and hom demand
    data = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'r')
    data_algo = data['algorithm']

    # Average the results for the hom demand over the number of sets
    bpp_hom, ba_hom = average_hom_demand(save=save, comp_folder=comp_folder,
                                         hom_pattern=hom_pattern,
                                         nbr_of_hom_sets=nbr_of_hom_sets)

    bpp_ed = data_algo['bpp'][()]
    ba_ed = data_algo['ba'][()]

    plot_comp_hom_demand(city=city, save=save, bpp_ed=bpp_ed,
                         bpp_hom=bpp_hom, ba_ed=ba_ed, ba_hom=ba_hom,
                         params=params, paths=paths)

    data.close()


def plot_dynamic_vs_static(save, paths, params):
    """
    Comparison between the dynamic approach and a static approach, without
    recalculation of the cyclists routes after each step.
    :param save: save name of the city
    :type save: str
    :param paths: Paths for loading and saving from/to.
    :type paths: dict
    :param params: Params for plotting
    :type params: dict
    :return:
    """
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]})

    comp_folder = f'{paths["comp_folder"]}'
    
    # Static
    data_static = h5py.File(f'{comp_folder}comp_{save}_static.hdf5', 'r')
    data_static_algo = data_static['algorithm']
    end_stat = data_static_algo['end'][()]
    bpp_stat = data_static_algo['bpp'][()]
    ba_stat = data_static_algo['ba'][()]
    los_stat = data_static_algo['los'][()]
    tfdt_stat = data_static_algo['tfdt'][()]
    data_static.close()
    
    # Static without penalty weighting for load calculation
    data_static_npw = h5py.File(f'{comp_folder}comp_{save}_static_npw.hdf5', 'r')
    data_static_npw_algo = data_static_npw['algorithm']
    end_npw = data_static_npw_algo['end'][()]
    bpp_npw = data_static_npw_algo['bpp'][()]
    ba_npw = data_static_npw_algo['ba'][()]
    los_npw = data_static_npw_algo['los'][()]
    tfdt_npw = data_static_npw_algo['tfdt'][()]
    
    # Dynamic
    data_dyn = h5py.File(f'{comp_folder}comp_{save}.hdf5', 'r')
    data_dyn_algo = data_dyn['algorithm']
    end_dyn = data_dyn_algo['end'][()]
    bpp_dyn = data_dyn_algo['bpp'][()]
    ba_dyn = data_dyn_algo['ba'][()]
    los_dyn = data_dyn_algo['los'][()]
    tfdt_dyn = data_dyn_algo['tfdt'][()]
    data_dyn.close()
    
    fig1, ax1 = plt.subplots(dpi=params["dpi"],
                             figsize=params["figs_hom_comp"])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    ax1.plot(bpp_dyn, ba_dyn, lw=params["lw_ed"], label='Dynamic', c='C0')
    ax1.plot(bpp_stat, ba_stat, lw=params["lw_ed"], label='Static Imp', c='C1')
    ax1.plot(bpp_npw, ba_npw, lw=params["lw_ed"], label='Static Cyc', c='C2')

    ax1.set_ylabel('bikeability $b(\lambda)$', fontsize=params["fs_axl"])
    ax1.tick_params(axis='y', labelsize=params["fs_ticks"], width=0.5)
    ax1.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    ax1.set_xlabel(r'normalized relative length of bike paths $\lambda$',
                   fontsize=params["fs_axl"])

    f_stat = interpolate.interp1d(bpp_stat, ba_stat)
    ba_sta_stat = f_stat(bpp_dyn)
    ba_rel_stat = [b / ba_sta_stat[m] for m, b in enumerate(ba_dyn)]
    f_npw = interpolate.interp1d(bpp_npw, ba_npw)
    ba_sta_npw = f_npw(bpp_dyn)
    ba_rel_npw = [b / ba_sta_npw[m] for m, b in enumerate(ba_dyn)]

    axins1 = inset_axes(ax1, width="50%", height="50%", loc=4, borderpad=.75)
    for axis in ['top', 'bottom', 'left', 'right']:
        axins1.spines[axis].set_linewidth(0.5)
    axins1.set_xlim(0.0, 1.0)
    axins1.set_ylim(0.9, 1.5)

    axins1.plot(bpp_dyn, ba_rel_stat, lw=params["lw_ed"], c='C1')
    axins1.plot(bpp_dyn, ba_rel_npw, lw=params["lw_ed"], c='C2')
    axins1.axhline(y=1, xmax=1, xmin=0, c='#808080', ls='--', lw=0.5)
    axins1.set_ylabel(r'relative bikeability $b(\lambda)$',
                      labelpad=1, fontsize=5)

    axins1.tick_params(axis='y', length=2, width=0.5, pad=0.5, labelsize=4)
    axins1.tick_params(axis='x', length=2, width=0.5, pad=0.5, labelsize=4)

    fig1.savefig(
        f'{paths["plot_folder"]}results/{save}/{save}_ba_comp'
        f'.{params["plot_format"]}', bbox_inches='tight')

    fig2, ax2 = plt.subplots(dpi=params["dpi"],
                             figsize=params["figs_hom_comp"])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(0.5)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(1.0, 1.5)

    df_dyn = [i / tfdt_dyn[end_dyn + 1] for i in tfdt_dyn[:end_dyn + 1]]
    ax2.plot(bpp_dyn, df_dyn, lw=params["lw_ed"], label='Dynamic', c='C0')
    df_stat = [i / tfdt_stat[end_stat + 1] for i in tfdt_stat[:end_stat + 1]]
    ax2.plot(bpp_stat, df_stat, lw=params["lw_ed"], label='Static Imp', c='C1')
    df_npw = [i / tfdt_npw[end_npw + 1] for i in tfdt_npw[:end_npw + 1]]
    ax2.plot(bpp_npw, df_npw, lw=params["lw_ed"], label='Static Cyc', c='C2')

    ax2.set_ylabel(r'perceived distance traveled $\mathcal{L}(\lambda)$',
                   fontsize=params["fs_axl"])
    ax2.set_xlabel(r'normalized relative length of bike paths $\lambda$',
                   fontsize=params["fs_axl"])

    ax2.tick_params(axis='y', labelsize=params["fs_ticks"], width=0.5)
    ax2.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())

    h_stat = interpolate.interp1d(bpp_stat, df_stat)
    df_sta_stat = h_stat(bpp_dyn)
    df_rel_stat = [b / df_sta_stat[m] for m, b in enumerate(df_dyn)]
    h_npw = interpolate.interp1d(bpp_npw, df_npw)
    df_sta_npw = h_npw(bpp_dyn)
    df_rel_npw = [b / df_sta_npw[m] for m, b in enumerate(df_dyn)]

    axins2 = inset_axes(ax2, width="50%", height="50%", loc=1, borderpad=.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        axins2.spines[axis].set_linewidth(0.5)
    axins2.set_xlim(0.0, 1.0)
    axins2.set_ylim(0.95, 1.01)

    axins2.plot(bpp_dyn, df_rel_stat, lw=params["lw_ed"], c='C1')
    axins2.plot(bpp_dyn, df_rel_npw, lw=params["lw_ed"], c='C2')
    axins2.axhline(y=1, xmax=1, xmin=0, c='#808080', ls='--', lw=0.5)
    axins2.set_ylabel('relative perceived distance', labelpad=1, fontsize=5)

    axins2.tick_params(axis='y', length=2, width=0.5, pad=0.5, labelsize=4)
    axins2.tick_params(axis='x', length=2, width=0.5, pad=0.5, labelsize=4)

    fig2.savefig(
        f'{paths["plot_folder"]}results/{save}/{save}_perceived_dist_comp'
        f'.{params["plot_format"]}', bbox_inches='tight')

    fig3, ax3 = plt.subplots(dpi=params["dpi"],
                             figsize=params["figs_hom_comp"])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax3.spines[axis].set_linewidth(0.5)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)

    ax3.plot(bpp_dyn, los_dyn, lw=params["lw_ed"], c='C0')
    ax3.plot(bpp_stat, los_stat, lw=params["lw_ed"], c='C1')
    ax3.plot(bpp_npw, los_npw, lw=params["lw_ed"], c='C2')

    ax3.set_ylabel('fraction traveled on street', fontsize=params["fs_axl"])
    ax3.tick_params(axis='y', labelsize=params["fs_ticks"], width=0.5)
    ax3.tick_params(axis='x', labelsize=params["fs_ticks"], width=0.5)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())

    ax3.set_xlabel(r'normalized relative length of bike paths $\lambda$',
                   fontsize=params["fs_axl"])
    ax3.set_title('Fraction on street', fontsize=params["fs_title"])

    fig3.savefig(
        f'{paths["plot_folder"]}results/{save}/{save}_len_on_street_comp'
        f'.{params["plot_format"]}', bbox_inches='tight')
