"""
This module includes all necessary helper functions for the data preparation and
handling.
"""

import json
import srtm
import warnings
import osmnx as ox
import networkx as nx
import pandas as pd
from ast import literal_eval
from shapely.geometry import Point, Polygon
from .setup_helper import create_default_params


def read_csv(path: str, delim: str = ",") -> pd.DataFrame:
    """Reads the csv given by path. Delimiter of csv can be chosen by delim.
    All column headers ar converted to lower case.

    Parameters
    ----------
    path : str
        Path to load csv from
    delim : str
        Delimiter of csv (Default value = ",")

    Returns
    -------
    output: pd.DataFrame

    """
    df = pd.read_csv(path, delimiter=delim)
    df.columns = map(str.lower, df.columns)
    return df


def write_csv(df: pd.DataFrame, path: str):
    """Writes given data frame to csv.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    path : str
        Path to save

    Returns
    -------

    """
    df.to_csv(path, index=False)


def load_demand(path: str) -> dict:
    return load_algorithm_results(path)


def load_algorithm_results(path: str) -> dict:
    """Load results created by the BikePathNet algorithm.

    Parameters
    ----------
    path : str
    Path of the data.

    Returns
    -------

    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_algorithm_results(data, save_path: str):
    """Save algorithm results.

    Parameters
    ----------
    data :
        Results
    save_path : str
        Filepath

    Returns
    -------

    """
    with open(save_path, "w") as fp:
        json.dump(data, fp)


def load_algorithm_params(path: str, params: dict) -> dict:
    """Load parameters used in the algorithm run and write them into params.

    Parameters
    ----------
    path : str
        Filepath
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------
    output: dict
        Params updated by algorithm params
    """
    with open(path, "r") as f:
        data = json.load(f)

    params["street_cost"] = data["street_cost"]
    params["car_penalty"] = data["car_penalty"]
    params["slope_penalty"] = data["slope_penalty"]
    params["turn_penalty"] = data["turn_penalty"]

    return params


def get_street_type(
    G: nx.MultiGraph | nx.MultiDiGraph,
    edge: tuple[int, int] | tuple[int, int, int],
    motorway: set | list | None = None,
    bike_paths: set | list | None = None,
) -> str:
    """Returns the street type of the edge. Street types are reduced to
    primary, secondary, tertiary and residential.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    edge : tuple[int, int] | tuple[int, int, int]
        Edge in the networkx graph.
    motorway : set | list | None
        Car only edges. (Default value = None)
    bike_paths : set | list | None
        Bike only edges. (Default value = None)

    Returns
    -------
    output : str

    """
    if motorway is None:
        motorway = ["motorway", "motorway_link"]
    if bike_paths is None:
        bike_paths = {"track", "service", "pedestrian", "cycleway", "busway", "path"}
    if "trunk" not in motorway:
        primary = ["primary", "primary_link", "trunk", "trunk_link"]
    else:
        primary = ["primary", "primary_link"]
    secondary = ["secondary", "secondary_link"]
    tertiary = ["tertiary", "tertiary_link"]

    street_type = G.edges[edge]["highway"]

    if isinstance(street_type, list):
        street_type_ranked = {
            st: (
                0
                if st in motorway
                else (
                    1
                    if st in primary
                    else (
                        2
                        if st in secondary
                        else 3 if st in tertiary else 5 if st in bike_paths else 4
                    )
                )
            )
            for st in street_type
        }
        street_type = sorted(street_type_ranked, key=street_type_ranked.get)[0]
    elif isinstance(street_type, str):
        street_type = street_type
    else:
        print(f"Street type unusable: {street_type}")

    if street_type in motorway:
        return "motorway"
    if street_type in primary:
        return "primary"
    elif street_type in secondary:
        return "secondary"
    elif street_type in tertiary:
        return "tertiary"
    elif street_type in bike_paths:
        return "bike_path"
    else:
        return "residential"


def get_lat_long_trips(
    path_to_trips: str, polygon: Polygon | None = None, delim: str = ","
) -> tuple[list, list, list, list, list]:
    """Returns five lists. The first stores the number of cyclists on this trip,
    the second the start latitude, the third the start longitude,
    the fourth the end latitude, the fifth the end longitude.
    An index corresponds to the same trip in each list.

    Parameters
    ----------
    path_to_trips : str
        path to the compacted trips csv.
    polygon : shapely.Polygon | None
        If only trips inside a polygon should be considered,
        pass it here. (Default value = None)
    delim : str
        Delimiter of the trips csv. (Default value = ",")

    Returns
    -------
    output : tuple[list, list, list, list, list]
        number of trips, start lat, start long, end lat, end long

    """
    trips = read_csv(path_to_trips, delim=delim)

    if polygon is None:
        start_lat = list(trips["start latitude"])
        start_long = list(trips["start longitude"])
        end_lat = list(trips["end latitude"])
        end_long = list(trips["end longitude"])
        nbr_of_trips = list(trips["number of trips"])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long
    else:
        trips["start in polygon"] = trips[["start latitude", "start longitude"]].apply(
            lambda row: polygon.intersects(
                Point(row["start longitude"], row["start latitude"])
            ),
            axis=1,
        )
        trips["end in polygon"] = trips[["end latitude", "end longitude"]].apply(
            lambda row: polygon.intersects(
                Point(row["end longitude"], row["end latitude"])
            ),
            axis=1,
        )
        trips["in polygon"] = trips[["start in polygon", "end in polygon"]].apply(
            lambda row: row["start in polygon"] and row["end in polygon"],
            axis=1,
        )
        start_lat = list(trips.loc[trips["in polygon"]]["start latitude"])
        start_long = list(trips.loc[trips["in polygon"]]["start longitude"])
        end_lat = list(trips.loc[trips["in polygon"]]["end latitude"])
        end_long = list(trips.loc[trips["in polygon"]]["end longitude"])
        nbr_of_trips = list(trips.loc[trips["in polygon"]]["number of trips"])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long


def get_bbox_of_trips(
    path_to_trips: str, polygon: Polygon | None = None, delim: str = ","
) -> tuple[float, float, float, float]:
    """Returns the bbox of the trips given by path_to_trips. If only trips inside
    a polygon should be considered, you can pass it to the polygon param.

    Parameters
    ----------
    path_to_trips : str
        path to the compacted trips csv.
    polygon : Shapely Polygon
        If only trips inside a polygon should be considered,
        pass it here. (Default value = None)
    delim : str
        Delimiter of the trips csv. (Default value = ',')

    Returns
    -------
    output: tuple[float, float, float, float]
        bbox [east, south, west, north]

    """
    trips_used, start_lat, start_long, end_lat, end_long = get_lat_long_trips(
        path_to_trips, polygon, delim=delim
    )
    north = max(start_lat + end_lat) + 0.005
    south = min(start_lat + end_lat) - 0.005
    east = max(start_long + end_long) + 0.01
    west = min(start_long + end_long) - 0.01
    return east, south, west, north


def load_trips(G, path_to_trips, polygon=None, delim=","):
    """Loads the trips and maps lat/long of start and end station to nodes in
    graph G. For this the ox.distance.nearest_nodes function of osmnx is used.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        graph used for lat/long to node mapping
    path_to_trips : str
        path to the trips csv.
    polygon : Shapely Polygon
        If only trips inside a polygon should be considered,
        pass it here. (Default value = None)
    delim : str
        Delimiter of the trips csv. (Default value = ")
    " :


    Returns
    -------
    type
        dict with trip info and set of stations used.
        trip_nbrs structure: key=(origin node, end node), value=# of cyclists

    """

    (
        nbr_of_trips,
        start_lat,
        start_long,
        end_lat,
        end_long,
    ) = get_lat_long_trips(path_to_trips, polygon, delim=delim)

    start_nodes = list(ox.nearest_nodes(G, start_long, start_lat))
    end_nodes = list(ox.nearest_nodes(G, end_long, end_lat))

    trip_nbrs = {}
    for trip in range(len(nbr_of_trips)):
        if (int(start_nodes[trip]), int(end_nodes[trip])) in trip_nbrs.keys():
            trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] += int(
                nbr_of_trips[trip]
            )
        else:
            trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] = int(
                nbr_of_trips[trip]
            )

    stations = set()
    for k, v in trip_nbrs.items():
        stations.add(k[0])
        stations.add(k[1])

    return trip_nbrs, stations


def trip_cyclist_type_split(trips: dict, cyclist_split: dict | None = None) -> dict:
    """Split demand along cyclist types.

    Parameters
    ----------
    trips : dict
        Demand
    cyclist_split : dict | None
        How to split the demand {cyclist_id: split factor}, split factors should
        sum to 1. (Default value = None)

    Returns
    -------
    output: dict
        Split demand.
    """
    if cyclist_split is None:
        cyclist_split = {1: 1}
    return {
        trip_od: {cyclist: trip_nbr * split for cyclist, split in cyclist_split.items()}
        for trip_od, trip_nbr in trips.items()
    }


def save_polygon_as_json(polygon: Polygon, save_path: str):
    """Save given polygon as json in geojson structure.

    Parameters
    ----------
    polygon : Polygon

    save_path : str

    Returns
    -------

    """
    polygon_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [list(polygon.exterior.coords)],
                },
            }
        ],
    }
    with open(save_path, "w") as fp:
        json.dump(polygon_data, fp)


def get_polygon_from_json(path_to_json: str) -> Polygon:
    """Reads json with single polygon. json can be created at http://geojson.io/.

    Parameters
    ----------
    path_to_json : str
        file path to json.

    Returns
    -------
    Polygon

    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    coordinates = [(item[0], item[1]) for item in coordinates]
    polygon = Polygon(coordinates)
    return polygon


def get_polygons_from_json(path_to_json: str) -> list[Polygon]:
    """Reads json with multiple polygons. json can be created at http://geojson.io/.

    Parameters
    ----------
    path_to_json : str
        file path to json.

    Returns
    -------
    output : list[Polygon]

    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    polygons = []
    for d in data["features"]:
        coordinates = d["geometry"]["coordinates"][0]
        coordinates = [(item[0], item[1]) for item in coordinates]
        polygons.append(Polygon(coordinates))
    return polygons


def get_polygon_from_bbox(bbox: tuple[float, float, float, float]) -> Polygon:
    """Returns the Polygon resembled by the given bbox.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        bbox [east, south, west, north]

    Returns
    -------
    output : Polygon

    """
    east, south, west, north = bbox
    corners = [(east, north), (west, north), (west, south), (east, south)]
    polygon = Polygon(corners)
    return polygon


def get_bbox_from_polygon(polygon: Polygon) -> tuple[float, float, float, float]:
    """Returns bbox from given polygon.

    Parameters
    ----------
    polygon : Polygon

    Returns
    -------
    output : tuple[float, float, float, float]
        bbox [east, south, west, north]

    """
    x, y = polygon.exterior.coords.xy
    points = [(i, y[j]) for j, i in enumerate(x)]

    west, south = float("inf"), float("inf")
    east, north = float("-inf"), float("-inf")
    for x, y in points:
        west = min(west, x)
        south = min(south, y)
        east = max(east, x)
        north = max(north, y)

    return east, south, west, north


def drop_invalid_values(
    csv: str,
    column: str,
    values: list,
    save: bool = False,
    save_path: str = "",
    delim: str = ",",
) -> pd.DataFrame:
    """Drops all rows if they have the given invalid value in the given column.

    Parameters
    ----------
    csv : str
        Path to csv.
    column : str
        Column name which should be checked vor invalid values.
    values : list
        Invalid values in the column.
    save : bool
        Set true if df should be saved as csv after dropping. (Default value = False)
    save_path : str
        Path where it should e saved. (Default value = ',')
    delim : str
        Delimiter of the original csv. (Default value = ',')

    Returns
    -------
    output : pd.DataFrame
        Data without the invalid values.

    """
    df = read_csv(csv, delim)
    for value in values:
        drop_ind = df[df[column] == value].index
        df.drop(drop_ind, inplace=True)
    if save:
        write_csv(df, save_path)
    return df


def consolidate_nodes(
    G: nx.MultiGraph | nx.MultiDiGraph, tol: float
) -> nx.MultiGraph | nx.MultiDiGraph:
    """Consolidates intersections of graph g with given tolerance in meters.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Graph to consolidate intersections in
    tol : float
        Tolerance for consolidation in meters

    Returns
    -------
    output : nx.MultiGraph | nx.MultiDiGraph

    """
    H = ox.project_graph(G)
    H = ox.consolidate_intersections(
        H,
        tolerance=tol,
        rebuild_graph=True,
        dead_ends=True,
        reconnect_edges=True,
    )
    print(
        f"Consolidating intersections. "
        f"Nodes before: {len(G.nodes)}. Nodes after: {len(H.nodes)}"
    )
    H = nx.convert_node_labels_to_integers(H)
    nx.set_node_attributes(H, {n: n for n in H.nodes}, "osmid")
    G = ox.project_graph(H, to_crs="epsg:4326")
    return G


def get_traffic_signal_nodes(
    G: nx.MultiGraph | nx.MultiDiGraph, tol: float = 10
) -> set[int]:
    """Returns a list with all nodes which have a traffic signal at the intersection
    with max. size 'tol'.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network
    tol : float
        Maximal distance [meter] of a node to the nearest node to form an intersection.

    Returns
    -------
    output : set[int]

    """
    H = ox.project_graph(G)
    H = ox.consolidate_intersections(
        H,
        tolerance=tol,
        rebuild_graph=True,
    )

    traffic_signal_list = []
    for n, d in H.nodes(data=True):
        if isinstance(d["osmid_original"], str):
            d["osmid_original"] = literal_eval(d["osmid_original"])
        elif isinstance(d["osmid_original"], list):
            d["osmid_original"] = d["osmid_original"]
        else:
            d["osmid_original"] = [int(d["osmid_original"])]
        for o_id in d["osmid_original"]:
            if (
                "highway" in G.nodes[o_id].keys()
                and G.nodes[o_id]["highway"] == "traffic_signals"
            ):
                traffic_signal_list += d["osmid_original"]

    return set(traffic_signal_list)


def check_existing_infrastructure(
    exinf: str | list[str], ex_inf_type: list[str] | None = None
) -> bool:
    """Check if the existing bike path/lane infrastructure meets the required
    type defined in ex_inf_type or not. Check OSM wiki for possible values.
    https://wiki.openstreetmap.org/wiki/Key:cycleway

    Parameters
    ----------
    exinf : str or list[str]
        Existing bike lane/path
    ex_inf_type : list[str]  or None
        List of infrastructure types which are accepted. (Default value = None)

    Returns
    -------
    output : bool
        Returns True if existing infrastructure type meets the
        requirements or False if not.

    """
    if ex_inf_type is None:
        ex_inf_type = ["track", "opposite_track"]
    if isinstance(exinf, str):
        check_exinf = exinf
    else:
        # If ex inf is not a str but a list, check if list is only made up
        # out of acceptable types.
        if set(exinf) == set(ex_inf_type):
            check_exinf = exinf[0]
        else:
            check_exinf = "no bike acceptable path or lane"
    if check_exinf in ex_inf_type:
        return True
    else:
        return False


def get_turn_angle(
    edge_in: tuple[int, int, int],
    edge_out: tuple[int, int, int],
    G: nx.MultiGraph | nx.MultiDiGraph,
) -> float:
    """Returns the turn angle between 'edge_in' and 'edge_out'.

    Parameters
    ----------
    edge_in : tuple[int, int, int]
        Ingoing edge
    edge_out : tuple[int, int, int]
        Outgoing edge
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network

    Returns
    -------
    output : float
    """
    e_in = G[edge_in[0]][edge_in[1]][edge_in[2]]
    e_out = G[edge_out[0]][edge_out[1]][edge_out[2]]

    bearing_in = e_in["bearing"]
    bearing_out = e_out["bearing"]

    if 0 <= bearing_in <= 180:
        if (180 + bearing_out) - bearing_in > 0:
            return bearing_in - bearing_out + 180
        else:
            return bearing_in - bearing_out - 180
    else:
        if (180 - bearing_out) + bearing_in > 0:
            return bearing_in - bearing_out + 180
        else:
            return bearing_in - bearing_out + 540


def get_turn_order(
    edge_out: tuple[int, int, int], G: nx.MultiGraph | nx.MultiDiGraph
) -> list[tuple[int, int, int]]:
    """Returns the order of all ingoing edges to the 'edge_out' from smallest to
    largest turn angle.

    Parameters
    ----------
    edge_out : tuple[int, int, int]

    G : nx.MultiGraph | nx.MultiDiGraph
        Street network

    Returns
    -------
    output: list[tuple[int, int, int]]
    """
    n = edge_out[0]
    in_neighbors = list(G.predecessors(n))
    in_edges = [(n_in, n, 0) for n_in in in_neighbors]
    turns = {edge_in: get_turn_angle(edge_in, edge_out, G) for edge_in in in_edges}

    return [k for k, v in sorted(turns.items(), key=lambda item: item[1])]


def set_intersection_size(
    G: nx.MultiGraph | nx.MultiDiGraph,
) -> nx.MultiGraph | nx.MultiDiGraph:
    """Sets the intersection size, defined by road size and existence of traffic
    signals.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Street network

    Returns
    -------
    output : nx.MultiGraph | nx.MultiDiGraph
    """
    for n, d in G.nodes(data=True):
        big_road = False
        mid_road = False
        for i in list(set(nx.all_neighbors(G, n))):
            s_type = get_street_type(G, (i, n, 0))
            if s_type in ["primary", "secondary"]:
                big_road = True
            if s_type in ["tertiary"]:
                mid_road = True

        if big_road and not d["traffic_signals"]:
            d["intersection_size"] = "large"
        elif mid_road and not d["traffic_signals"]:
            d["intersection_size"] = "medium"
        else:
            d["intersection_size"] = "small"

    return G


def set_turn_penalty(
    G: nx.MultiGraph | nx.MultiDiGraph,
) -> nx.MultiGraph | nx.MultiDiGraph:
    """Sets the turn penalty according to intersections size and if a large streets has
    to be crossed.

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph

    Returns
    -------
    output : nx.MultiGraph | nx.MultiDiGraph
    """
    turn_penalty = dict()
    for edge_out in G.edges(keys=True):
        n = edge_out[0]
        turn_order = get_turn_order(edge_out, G)

        turn_penalty_edge = dict()
        crossing_large_street = False
        for edge_in in turn_order:
            edge_in_s_type = get_street_type(G, edge_in)
            if crossing_large_street:
                turn_penalty_edge[edge_in[0]] = G.nodes[n]["intersection_size"]
            else:
                turn_penalty_edge[edge_in[0]] = "small"
            if edge_in_s_type in ["primary", "secondary"]:
                crossing_large_street = True

        turn_penalty[edge_out] = turn_penalty_edge
    nx.set_edge_attributes(G, turn_penalty, "turn_penalty")
    return G


def prepare_downloaded_map(
    G: nx.MultiGraph | nx.MultiDiGraph,
    trunk: bool = False,
    consolidate: bool = False,
    intersection_tol: float = 35,
    ex_inf: list[str] | None = None,
) -> nx.MultiDiGraph:
    """Prepares the downloaded map. Removes all motorway edges and if trunk=False also
    all trunk edges. Turns it to undirected, removes all isolated nodes using
    networkxs isolates() function and reduces the graph to the greatest connected
    component using osmnxs get_largest_component().

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Graph to clean.
    trunk : bool
        Decides if trunk should be kept or not. If you want to keep trunk in the graph,
        set to True. (Default value = False)
    consolidate : bool
        Set true if intersections should bes consolidated. (Default value = False)
    intersection_tol : float
        Tolerance of intersection consolidation in meters (Default value = 35)
    ex_inf : list[str] or None
        Which cycleways should be accepted as ex inf (Default value = None)

    Returns
    -------
    output : nx.MultiDiGraph

    """
    if ex_inf is None:
        ex_inf = ["track", "opposite_track"]

    # Check where bike paths/lanes already exist
    print("Check for existing infrastructure.")
    for u, v, k, d in G.edges(keys=True, data=True):
        if "cycleway" in d.keys() and check_existing_infrastructure(
            d["cycleway"], ex_inf
        ):
            d["ex_inf"] = True
        elif "cycleway:right" in d.keys() and check_existing_infrastructure(
            d["cycleway:right"], ex_inf
        ):
            d["ex_inf"] = True
        elif "cycleway:left" in d.keys() and check_existing_infrastructure(
            d["cycleway:left"], ex_inf
        ):
            d["ex_inf"] = True
        elif "bicycle" in d.keys() and d["bicycle"] == "use_sidepath":
            d["ex_inf"] = True
        elif (
            "sidewalk:both:bicycle" in d.keys() and d["sidewalk:both:bicycle"] == "yes"
        ):
            d["ex_inf"] = True
        elif (
            "sidewalk:right:bicycle" in d.keys()
            and d["sidewalk:right:bicycle"] == "yes"
        ):
            d["ex_inf"] = True
        elif (
            "sidewalk:left:bicycle" in d.keys() and d["sidewalk:left:bicycle"] == "yes"
        ):
            d["ex_inf"] = True
        elif (
            "cycleway:both:bicycle" in d.keys()
            and d["cycleway:both:bicycle"] == "designated"
        ):
            d["ex_inf"] = True
        elif (
            "cycleway:right:bicycle" in d.keys()
            and d["cycleway:right:bicycle"] == "designated"
        ):
            d["ex_inf"] = True
        elif (
            "cycleway:left:bicycle" in d.keys()
            and d["cycleway:left:bicycle"] == "designated"
        ):
            d["ex_inf"] = True
        else:
            d["ex_inf"] = False

    # Remove motorways and trunks
    if trunk:
        s_t = ["motorway", "motorway_link"]
    else:
        s_t = ["motorway", "motorway_link", "trunk", "trunk_link"]
    edges_to_remove = [
        e for e in G.edges(keys=True) if get_street_type(G, e, motorway=s_t) in s_t
    ]
    G.remove_edges_from(edges_to_remove)
    print(f"Removed {len(edges_to_remove)} car only edges.")

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes.")

    traffic_signal_nodes = get_traffic_signal_nodes(G)

    # Simplify graph, remove irrelevant nodes
    G = ox.simplify_graph(G)

    if consolidate:
        G = consolidate_nodes(G, intersection_tol)

    srtm_data = srtm.Srtm1HeightMapCollection()
    for n, d in G.nodes(data=True):
        lat = d["y"]
        lng = d["x"]
        try:
            d["elevation"] = srtm_data.get_altitude(latitude=lat, longitude=lng)
        except srtm.exceptions.NoHeightMapDataException:
            warnings.warn(
                f"SRTM directory '{srtm_data.hgt_dir}' is missing files. "
                f"Setting height 0 for all nodes."
            )
            d["elevation"] = 0
        if consolidate:
            if isinstance(d["osmid_original"], str):
                node_set = set(literal_eval(d["osmid_original"]))
            elif isinstance(d["osmid_original"], int):
                node_set = {d["osmid_original"]}
            else:
                node_set = set(d["osmid_original"])
            if len(node_set.intersection(traffic_signal_nodes)) != 0:
                d["traffic_signals"] = True
            else:
                d["traffic_signals"] = False
        else:
            if "highway" in d.keys() and d["highway"] == "traffic_signals":
                d["traffic_signals"] = True
            else:
                d["traffic_signals"] = False

    G = nx.MultiDiGraph(ox.convert.to_digraph(G))
    G = nx.MultiDiGraph(ox.convert.to_undirected(G))

    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print(f"Removed {len(self_loops)} self loops.")

    G = ox.truncate.largest_component(G, strongly=True)
    print("Reduce to largest connected component")

    G = nx.convert_node_labels_to_integers(G, first_label=1, ordering="default")

    # Calculate slope in percent
    print("Adding street gradients.")
    G = ox.add_edge_grades(G, add_absolute=True)

    # Set intersection penalties
    print("Setting intersection penalties.")
    G = ox.add_edge_bearings(G)
    G = set_intersection_size(G)
    G = set_turn_penalty(G)

    for u, v, k, d in G.edges(keys=True, data=True):
        d["highway"] = get_street_type(G, (u, v, k))
        if "ex_inf" in d.keys():
            if isinstance(d["ex_inf"], list):
                if True in d["ex_inf"]:
                    d["ex_inf"] = True
                else:
                    d["ex_inf"] = False
        elif d["highway"] == "bike_path":
            d["ex_inf"] = True
        else:
            d["ex_inf"] = False
        d["cost"] = d["length"]

    return G


def download_map_by_bbox(
    bbox: tuple[float, float, float, float],
    network_type: str = "drive",
    trunk: bool = False,
    consolidate: bool = False,
    tol: float = 35,
    truncate_by_edge: bool = False,
    params: dict | None = None,
) -> nx.MultiDiGraph:
    """Downloads a drive graph from osm given by the bbox and cleans it for usage.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Boundary box of the map [east, south, west, north].
    network_type : str
        Network type for download, see ox.graph_from_bbox (Default value = "drive")
    trunk : bool
        Decides if trunk should be kept or not. If you want to
        keep trunk in the graph, set to True. (Default value = False)
    consolidate : bool
        Set true if intersections should bes consolidated. (Default value = False)
    tol : float
        Tolerance of intersection consolidation in meters (Default value = 35)
    truncate_by_edge : bool
        if True, retain node if it’s outside bounding box but at least one of node’s
        neighbors are within bounding box (Default value = False)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------
    output : nx.MultiDiGraph

    """
    if params is None:
        params = create_default_params()

    print(
        f"Downloading map from bounding box. "
        f"Eastern bound: {bbox[0]}, "
        f"southern bound: {bbox[1]}, "
        f"western bound: {bbox[2]}, "
        f"northern bound: {bbox[3]}"
    )
    useful_tags = (
        ox.settings.useful_tags_way
        + ["cycleway"]
        + ["cycleway:right"]
        + ["cycleway:left"]
        + ["bicycle"]
        + ["sidewalk:both:bicycle"]
        + ["sidewalk:right:bicycle"]
        + ["sidewalk:left:bicycle"]
        + ["cycleway:both:bicycle"]
        + ["cycleway:right:bicycle"]
        + ["cycleway:left:bicycle"]
    )
    ox.settings.useful_tags_way = useful_tags
    G = ox.graph_from_bbox(
        bbox,
        network_type=network_type,
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, trunk, consolidate=consolidate, intersection_tol=tol, ex_inf=params["ex inf"]
    )

    return G


def download_map_by_name(
    city: str,
    nominatim_result: int = 1,
    network_type="drive",
    trunk: bool = False,
    consolidate: bool = False,
    tol: float = 35,
    truncate_by_edge: bool = False,
    params: dict | None = None,
) -> nx.MultiDiGraph:
    """Downloads a drive graph from osm given by the name and geocode of the nominatim
    database and  cleans it for usage.

    Parameters
    ----------
    city : str
        Name of the place to download.
    nominatim_result : int
        Which result of the nominatim database should be downloaded. (Default value = 1)
    network_type : str
        Network type for download, see ox.graph_from_bbox (Default value = "drive")
    trunk : bool
        Decides if trunk should be kept or not. If you want to keep trunk in the graph,
        set to True. (Default value = False)
    consolidate : bool
        Set true if intersections should bes consolidated. (Default value = False)
    tol : float
        Tolerance of intersection consolidation in meters (Default value = 35)
    truncate_by_edge : bool
        if True, retain node if it’s outside the area but at least one of node’s
        neighbors are within the area (Default value = False)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------
    output : nx.MultiDiGraph

    """
    if params is None:
        params = create_default_params()

    print(
        f"Downloading map by place. Name of the place: {city}, "
        f"Nominatim result number {nominatim_result}."
    )
    useful_tags = (
        ox.settings.useful_tags_way
        + ["cycleway"]
        + ["cycleway:right"]
        + ["cycleway:left"]
        + ["bicycle"]
        + ["sidewalk:both:bicycle"]
        + ["sidewalk:right:bicycle"]
        + ["sidewalk:left:bicycle"]
        + ["cycleway:both:bicycle"]
        + ["cycleway:right:bicycle"]
        + ["cycleway:left:bicycle"]
    )
    ox.settings.useful_tags_way = useful_tags
    G = ox.graph_from_place(
        city,
        which_result=nominatim_result,
        network_type=network_type,
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, trunk, consolidate=consolidate, intersection_tol=tol, ex_inf=params["ex inf"]
    )

    return G


def download_map_by_polygon(
    polygon,
    network_type="drive",
    trunk=False,
    consolidate=False,
    tol=35,
    truncate_by_edge=False,
    params=None,
) -> nx.MultiDiGraph:
    """Downloads a drive graph from osm given by the polygon and cleans it for
    usage.

    Parameters
    ----------
    polygon : Polygon
        Polygon of the area.
    network_type : str
        Network type for download, see ox.graph_from_bbox (Default value = "drive")
    trunk : bool
        Decides if trunk should be kept or not. If you want to keep trunk in the graph,
        set to True. (Default value = False)
    consolidate : bool
        Set true if intersections should bes consolidated. (Default value = False)
    tol : float
        Tolerance of intersection consolidation in meters (Default value = 35)
    truncate_by_edge : bool
        if True, retain node if it’s outside bounding box but at least one of node’s
        neighbors are within bounding box (Default value = False)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)

    Returns
    -------
    output :  nx.MultiDiGraph

    """
    if params is None:
        params = create_default_params()

    print("Downloading map by polygon.")
    useful_tags = (
        ox.settings.useful_tags_way
        + ["cycleway"]
        + ["cycleway:right"]
        + ["cycleway:left"]
        + ["bicycle"]
        + ["sidewalk:both:bicycle"]
        + ["sidewalk:right:bicycle"]
        + ["sidewalk:left:bicycle"]
        + ["cycleway:both:bicycle"]
        + ["cycleway:right:bicycle"]
        + ["cycleway:left:bicycle"]
    )
    ox.settings.useful_tags_way = useful_tags
    G = ox.graph_from_polygon(
        polygon,
        network_type=network_type,
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, trunk, consolidate=consolidate, intersection_tol=tol, ex_inf=params["ex inf"]
    )

    return G


def save_demand(trips: dict, file_path: str):
    """Saves demand to given path.

    Parameters
    ----------
    trips : dict
        Demand
    file_path : str

    Returns
    -------

    """
    with open(file_path, "w") as fp:
        json.dump({str(k): v for k, v in trips.items()}, fp)
