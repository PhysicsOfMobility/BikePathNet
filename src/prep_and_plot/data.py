"""
This module includes all necessary functions for the data preparation and handling.
"""
from os.path import join
from pathlib import Path
from shutil import copyfile
from .data_helper import *
from .plot import plot_used_nodes, plot_street_network
from .setup_helper import create_default_paths


def prep_city(
    city_name: str,
    save_name: str,
    input_csv: str,
    polygon_file: str | None = None,
    network_type: str = "drive",
    nominatim_name: str | None = None,
    nominatim_result: int = 1,
    car_only_street_types: list[str] | None = None,
    ensure_highway_diff: bool = False,
    consolidate: bool = False,
    intersection_tol: float = 35,
    remove_dead_ends: bool = False,
    dead_end_tol: float = 50,
    by_bbox: bool = False,
    by_city: bool = False,
    by_polygon: bool = True,
    cached_graph: bool = False,
    cached_graph_file_path: str | None = None,
    params: dict | None = None,
    paths: dict | None = None,
):
    """Prepares the data of a city for the algorithm and saves it to the desired
    location.

    Parameters
    ----------
    city_name : str
        Name of the city
    save_name : str
        Save name of the city
    input_csv : str
        Path to the trip csv
    polygon_file : str | None
        Path to the polygon file. (Default value = None)
    network_type : str
        Type of the osmnx network (e.g. drive or bike) (Default value = "drive")
    nominatim_name : str
        Nominatim name of the city (Default value = None)
    nominatim_result : int
        results position of the city for the given name (Default value = 1)
    car_only_street_types : list[str] | None
        List of generally car-only street types to keep (Default value = ["motorway", "trunk"])
    ensure_highway_diff : bool
        Set true if graph simplification should ensure all highway tags. (Default value = False)
    consolidate : bool
        Set true if intersections should be consolidated. (Default value = False)
    intersection_tol : float
        Tolerance of intersection consolidation in meters (Default value = 35)
    remove_dead_ends : bool
        Set true if (short) dead-ends should be removed. (Default value = False)
    dead_end_tol : float
        Maximal length of removed dead-ends in meters (Default value = 50)
    by_bbox : bool
        If graph should be downloaded by the bbox surrounding the
        trips (Default value = True)
    by_city : bool
        If graph should be downloaded by the nominatim name (Default value = True)
    by_polygon : bool
        If graph should be downloaded by the given polygon (Default value = True)
    cached_graph : bool
        If a previously downloaded graph should be used. (Default value = False)
    cached_graph_file_path : str or None
        Path of the downloaded graph. (Default value = None)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict | None
        Dict with the paths for data, plots etc.

    Returns
    -------

    """
    if polygon_file is None:
        polygon_file = join(paths["polygon_folder"], f"{save_name}.json")
    if paths is None:
        paths = create_default_paths()
    if params is None:
        params = create_default_params()

    output_folder = join(paths["input_folder"], save_name)
    plot_folder = join(paths["plot_folder"], "preparation", save_name)

    # Check if necessary folders exists, otherwise create.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    if nominatim_name is None:
        nominatim_name = city_name
    if cached_graph_file_path is None:
        cached_graph_file_path = join(output_folder, f"{save_name}.graphml")

    print(f"Preparing {city_name}.")

    if by_bbox:
        # Get bounding box of trips
        print("Getting bbox of trips.")
        bbox = get_bbox_of_trips(input_csv)

        if not cached_graph:
            # Download map given by bbox
            print("Downloading map given by bbox.")
            G_b = download_map_by_bbox(
                bbox,
                network_type=network_type,
                car_only_street_types=car_only_street_types,
                ensure_highway_diff=ensure_highway_diff,
                consolidate=consolidate,
                intersection_tol=intersection_tol,
                remove_dead_ends=remove_dead_ends,
                dead_end_tol=dead_end_tol,
                params=params,
                paths=paths,
            )
        else:
            print("Loading cached bbox map")
            G_b = ox.load_graphml(filepath=cached_graph_file_path)
            for u, v, data in G_b.edges(data=True):
                data["ex_inf"] = literal_eval(data["ex_inf"])
                data["turn_penalty"] = {
                    int(key): value for key, value in data["turn_penalty"].items()
                }
                data["cost"] = literal_eval(data["cost"])
            if consolidate:
                G_b = consolidate_nodes(G_b, tol=intersection_tol)
            G_b = nx.convert_node_labels_to_integers(G_b, first_label=1, ordering="default")

        # Loading trips inside bbox
        print("Mapping stations and calculation trips on map given by bbox")
        trips_b, stations_b = load_trips(G_b, input_csv)
        trips_b = trip_cyclist_type_split(
            trips_b, cyclist_split=params["cyclist_split"]
        )
        print(
            f"Number of stations: {len(stations_b)}, "
            f"Number of trips: {sum([sum(t_b.values()) for t_b in trips_b.values()]):3.2f}, "
            f"Unique trips: {sum([len(t_b.keys()) for t_b in trips_b.values()])} "
        )

        # Colour all nodes used as origin or destination
        print("Plotting used nodes on graph given by bbox.")
        plot_used_nodes(
            city=city_name,
            save=f"{save_name}_bbox",
            G=G_b,
            trip_nbrs=trips_b,
            stations=stations_b,
            plot_folder=plot_folder,
            params=params,
        )
        # Plot used street network with all nodes
        plot_street_network(
            city=city_name,
            save=f"{save_name}_bbox",
            G=G_b,
            plot_folder=plot_folder,
            params=params,
        )

        ox.save_graphml(G_b, join(output_folder, f"{save_name}_bbox.graphml"))
        save_demand(trips_b, join(output_folder, f"{save_name}_bbox_demand.json"))

    if by_city:
        if not cached_graph:
            # Download whole map of the city
            print("Downloading complete map of city")
            G_c = download_map_by_name(
                nominatim_name,
                nominatim_result,
                network_type=network_type,
                car_only_street_types=car_only_street_types,
                ensure_highway_diff=ensure_highway_diff,
                consolidate=consolidate,
                intersection_tol=intersection_tol,
                remove_dead_ends=remove_dead_ends,
                dead_end_tol=dead_end_tol,
                params=params,
            )
        else:
            print("Loading cached map of city")
            G_c = ox.load_graphml(filepath=cached_graph_file_path)
            for u, v, data in G_c.edges(data=True):
                data["ex_inf"] = literal_eval(data["ex_inf"])
                data["turn_penalty"] = {
                    int(key): value for key, value in data["turn_penalty"].items()
                }
                data["cost"] = literal_eval(data["cost"])
            if consolidate:
                G_c = consolidate_nodes(G_c, tol=intersection_tol)
            G_c = nx.convert_node_labels_to_integers(G_c, first_label=1, ordering="default")

        # Loading trips inside whole map
        print("Mapping stations and calculation trips on city map.")
        polygon_c = ox.geocode_to_gdf(nominatim_name, which_result=nominatim_result)
        trips_c, stations_c = load_trips(G_c, input_csv, polygon=polygon_c)
        trips_c = trip_cyclist_type_split(
            trips_c, cyclist_split=params["cyclist_split"]
        )
        print(
            f"Number of stations: {len(stations_c)}, "
            f"Number of trips: {sum([sum(t_c.values()) for t_c in trips_c.values()]):3.2f} "
            f"Unique trips: {sum([len(t_c.keys()) for t_c in trips_c.values()])} "
        )

        # Colour all nodes used as origin or destination
        print("Plotting used nodes on complete city.")
        plot_used_nodes(
            city=city_name,
            save=f"{save_name}_city",
            G=G_c,
            trip_nbrs=trips_c,
            stations=stations_c,
            plot_folder=plot_folder,
            params=params,
        )
        # Plot used street network with all nodes
        plot_street_network(
            city=city_name,
            save=f"{save_name}_city",
            G=G_c,
            plot_folder=plot_folder,
            params=params,
        )

        ox.save_graphml(G_c, join(output_folder, f"{save_name}_city.graphml"))
        save_demand(trips_c, join(output_folder, f"{save_name}_city_demand.json"))

    if by_polygon:
        # Download cropped map (polygon)
        polygon = get_polygon_from_json(polygon_file)

        if not cached_graph:
            print("Downloading polygon.")
            G = download_map_by_polygon(
                polygon,
                network_type=network_type,
                car_only_street_types=car_only_street_types,
                ensure_highway_diff=ensure_highway_diff,
                consolidate=consolidate,
                intersection_tol=intersection_tol,
                remove_dead_ends=remove_dead_ends,
                dead_end_tol=dead_end_tol,
                params=params,
            )
        else:
            print("Loading cached map.")
            G = ox.load_graphml(filepath=cached_graph_file_path)
            for u, v, data in G.edges(data=True):
                data["ex_inf"] = literal_eval(data["ex_inf"])
                data["turn_penalty"] = {
                    int(key): value for key, value in data["turn_penalty"].items()
                }
                data["cost"] = literal_eval(data["cost"])
            if consolidate:
                G = consolidate_nodes(G, tol=intersection_tol)
            G = nx.convert_node_labels_to_integers(G, first_label=1, ordering="default")

        # Loading trips inside the polygon
        print("Mapping stations and calculation trips in polygon.")
        trips, stations = load_trips(G, input_csv, polygon=polygon)
        trips = trip_cyclist_type_split(trips, cyclist_split=params["cyclist_split"])
        print(
            f"Number of stations: {len(stations)}, "
            f"Number of trips: {sum([sum(t.values()) for t in trips.values()]):3.2f}, "
            f"Unique trips: {sum([len(t.keys()) for t in trips.values()])}"
        )

        # Colour all nodes used as origin or destination
        print("Plotting used nodes in polygon.")
        plot_used_nodes(
            city=city_name,
            save=save_name,
            G=G,
            trip_nbrs=trips,
            stations=stations,
            plot_folder=plot_folder,
            params=params,
        )
        # Plot used street network with all nodes
        plot_street_network(
            city=city_name,
            save=f"{save_name}",
            G=G,
            plot_folder=plot_folder,
            params=params,
        )
        ox.save_graphml(G, join(output_folder, f"{save_name}.graphml"))
        save_demand(trips, join(output_folder, f"{save_name}_demand.json"))

def triangular_lattice_graph(n: int, m: int, bike_highways=None, central_node=None, shift: bool = False):
    """
    Parameters
    ----------
    n : int
        Number of columns of triangles.
    m : int
        Number of rows of triangles.
    bike_highways
    central_node
    shift

    Returns
    -------
    output : tuple[nx.MultiDiGraph, shp.geometry.polygon.Polygon]
        Triangular lattice graph and polygon of the area of the graph.
    """
    if bike_highways is None:
        bike_highways = [2]
    g = nx.triangular_lattice_graph(n, m, periodic=False, with_positions=True,  create_using=nx.Graph)

    bc = nx.betweenness_centrality(g)
    sp = nx.shortest_path(g, source=max(bc, key=bc.get))
    del_nodes = []
    for v, d in g.nodes(data=True):
        if len(sp[v]) - 1 > n / 2:
            del_nodes.append(v)
    g.remove_nodes_from(del_nodes)
    nx.relabel_nodes(g, {n: idx for idx, n in enumerate(g.nodes)}, copy=False)

    max_x = max([pos[0] for v, pos in g.nodes(data="pos")])
    max_y = max([pos[1] for v, pos in g.nodes(data="pos")])

    nx.set_node_attributes(g, {v: pos[0] - max_x / 2 for v, pos in
                               g.nodes(data="pos")}, "x")
    nx.set_node_attributes(g, {v: pos[1] - max_y / 2 for v, pos in
                               g.nodes(data="pos")}, "y")


    nx.set_edge_attributes(g, {e: 1 for e in g.edges()}, "length")
    nx.set_edge_attributes(g, {e: False for e in g.edges()}, "bike_highway")
    nx.set_edge_attributes(g, {e: False for e in g.edges()}, "blocked")
    nx.set_node_attributes(g, {n: False for n in g.nodes()}, "bike_highway")
    nx.set_node_attributes(g, {n: 0 for n in g.nodes()}, "level")

    for v, d in g.nodes(data=True):
        for k in list(d.keys()):
            if k in ["pos"]:
                del g.nodes[v][k]

    if central_node is None:
        bc = nx.betweenness_centrality(g)
        central_node = max(bc, key=bc.get)
    triangular_bike_highways(g, central_node=central_node, lengths=bike_highways)
    if shift:
        for highway_length in bike_highways:
            for i in range(1, highway_length):
                shift_nodes = [s_n for s_n in get_k_neighbours(g, central_node, i) if g.nodes[s_n]["x"] > 0.0 and abs(g.nodes[s_n]["y"]) <= np.tan(np.pi/3)*g.nodes[s_n]["x"]]
                for v in shift_nodes:
                    if not g.nodes[v]["bike_highway"]:
                        triangular_bike_highways(g, central_node=v, lengths=[highway_length])

    for u, v, d in g.edges(data=True):
        if "ramp" not in d.keys():
            d["ramp"] = False

    nx.set_edge_attributes(g, {(u, v): d for u, v, d in g.edges(data="length")}, "cost")

    g.remove_nodes_from(list(nx.isolates(g)))

    bpp_layers = set([d for n, d in g.nodes(data="level")])
    layer_edges = dict()
    layer_nodes = dict()
    for layer in bpp_layers:
        h = deepcopy(g)
        h.remove_nodes_from([n for n, d in g.nodes(data=True) if not d["level"] == layer])
        layer_edges[layer] = len(h.edges)
        layer_nodes[layer] = len(h.nodes)

    print(f"{n}-hex-grid with layers: {bpp_layers} \n"
          f"Total: nodes: {nx.number_of_nodes(g)}, edges {nx.number_of_edges(g)} \n"
          f"Layer: nodes: {layer_nodes} edges: {layer_edges}")

    g = nx.MultiDiGraph(g)
    g = nx.convert_node_labels_to_integers(g, first_label=1, label_attribute="old_node_label")
    old2new = {n_old: n_new for n_new, n_old in g.nodes(data="old_node_label")}
    for n, d in g.nodes(data=True):
        if "rampnode" in d.keys():
            d["rampnode"] = old2new[d["rampnode"]]
    g = add_osmnx_data(g)

    coords = [(d["x"], d["y"]) for v, d in g.nodes(data=True)]
    poly = polygon_from_coordinates(coords)

    for u, v, d in g.edges(data=True):
        if "ramp" not in d.keys():
            print((u, v))

    return g, poly


def gen_noisy_demand(
    clean_demand: dict,
    delta: float,
    save: str,
    noisy_demand_save: str,
    nodes_noise = None,
    rng: np.random.Generator = None,
    paths: dict | None = None,
    ):
    if paths is None:
        paths = create_default_paths()

    noisy_demand_folder = join(paths["input_folder"], noisy_demand_save)
    Path(noisy_demand_folder).mkdir(parents=True, exist_ok=True)

    demand_noisy, nodes_noise = noisy_demand(clean_demand, delta, nodes_noise=nodes_noise, rng=rng)
    with open(join(noisy_demand_folder, f"{noisy_demand_save}_demand.json"), "w") as fp:
        json.dump({str(k): {int(v_k): v_v for v_k, v_v in v.items()} for k, v in demand_noisy.items()}, fp)
    with open(join(noisy_demand_folder, f"{noisy_demand_save}_node_noise.json"), "w") as fp:
        json.dump(nodes_noise, fp)

    copyfile(join(paths["input_folder"], save, f"{save}.graphml"),
             join(noisy_demand_folder, f"{noisy_demand_save}.graphml"))
    copyfile(join(paths["input_folder"], save, f"{save}_penalties.json"),
             join(noisy_demand_folder, f"{noisy_demand_save}_penalties.json"))


def gen_noisy_penalty(
    penalties_dict: dict,
    penalty: dict,
    delta: float,
    save: str,
    noisy_penalty_save: str,
    min_penalty: float = 0.0,
    max_penalty: float = 1.0,
    rng: np.random.Generator = None,
    paths: dict | None = None,
    ):
    if paths is None:
        paths = create_default_paths()

    noisy_penalty_folder = join(paths["input_folder"], noisy_penalty_save)
    Path(noisy_penalty_folder).mkdir(parents=True, exist_ok=True)

    penalties_dict_noisy = deepcopy(penalties_dict)

    penalty_noisy = noisy_penalty(penalties_dict_noisy[penalty], delta, min_penalty=min_penalty, max_penalty=max_penalty, rng=rng)
    penalties_dict_noisy[penalty] = penalty_noisy

    with open(join(noisy_penalty_folder, f"{noisy_penalty_save}_penalties.json"), "w") as fp:
        json.dump(penalties_dict_noisy, fp)

    copyfile(join(paths["input_folder"], save, f"{save}.graphml"),
             join(noisy_penalty_folder, f"{noisy_penalty_save}.graphml"))
    copyfile(join(paths["input_folder"], save, f"{save}_demand.json"),
             join(noisy_penalty_folder, f"{noisy_penalty_save}_demand.json"))


def save_data(g, demand, poly, save, g_demand_folder, poly_folder):
    """
    Saves g and demand in the "g_demand_folder" and the poly in the "poly_folder". If these folders do not exist, they and all necessary parent folders are created.

    Parameters
    ----------
    g : nx.MultiDiGraph
    demand : dict[tuple[int, int], dict[int, float]]
    poly : shp.geometry.polygon.Polygon
    save : str
    g_demand_folder : str
    poly_folder : str

    Returns
    -------

    """
    Path(g_demand_folder).mkdir(parents=True, exist_ok=True)

    Path(poly_folder).mkdir(parents=True, exist_ok=True)

    ox.save_graphml(g, filepath=join(g_demand_folder, f"{save}.graphml"))

    with open(join(g_demand_folder, f"{save}_demand.json"), "w") as fp:
        json.dump({str(k): {int(v_k): v_v for v_k, v_v in v.items()} for k, v in demand.items()}, fp)

    save_polygon_as_json(poly, join(poly_folder, f"{save}.json"))