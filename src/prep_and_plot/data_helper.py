"""
This module includes all necessary helper functions for the data preparation and
handling.
"""
import bisect
import json
import rasterio
import srtm
import warnings

import itertools as it
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from ast import literal_eval
from copy import deepcopy
from shapely.geometry import LineString, Point, Polygon, MultiPoint
from .setup_helper import create_default_params, create_default_paths


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
    """
    Load demand data.

    Parameters
    ----------
    path : str
    Path of the data.

    Returns
    -------
    output: dict
    """
    demand = load_algorithm_results(path)
    return {literal_eval(k): v for k, v in demand.items()}


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


def load_algorithm_results(path: str) -> dict:
    """Load results created by the BikePathNet algorithm.

    Parameters
    ----------
    path : str
    Path of the data.

    Returns
    -------
    output: dict
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
        Car only street types. (Default value = ["motorway", "trunk"])
    bike_paths : set | list | None
        Bike only edges. (Default value = None)

    Returns
    -------
    output : str

    """
    if motorway is None:
        motorway = ["motorway", "motorway_link", "trunk", "trunk_link"]
    if "trunk" not in motorway:
        primary = ["primary", "primary_link", "trunk", "trunk_link"]
    else:
        primary = ["primary", "primary_link"]
    secondary = ["secondary", "secondary_link"]
    tertiary = ["tertiary", "tertiary_link"]
    if bike_paths is None:
        bike_paths = ["track", "service", "pedestrian", "cycleway", "path", "bike_path"]

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
            trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] += (
                nbr_of_trips)[trip]
        else:
            trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] = (
                nbr_of_trips)[trip]

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


def polygon_from_coordinates(coordinates):
    """
    Creates a polygon of the convex hull of the given coordinates.

    Parameters
    ----------
    coordinates :  numpy.ndarray | list[tuple[float, float]]

    Returns
    -------
    output : shapely.geometry.polygon.Polygon
        Polygon of the convex hull of the given coordinates.
    """
    pts = MultiPoint(coordinates)
    return pts.convex_hull


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


def add_elevation_data( G: nx.MultiGraph | nx.MultiDiGraph, raster_file: str | None = None):
    if raster_file is None:
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
    else:
        raster_crs = rasterio.open(raster_file).crs.to_epsg()
        H = ox.project_graph(G, to_crs=raster_crs)
        H = ox.add_node_elevations_raster(H, raster_file)
        G = ox.project_graph(H, to_crs="epsg:4326")
        for _, d in G.nodes(data=True):
            if d.get("elevation", None) is None:
                d["elevation"] = 0
    return G


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
    print(f"Consolidating intersections from {len(G.nodes)} to {len(H.nodes)} nodes.")
    H = nx.convert_node_labels_to_integers(H)
    nx.set_node_attributes(H, {n: n for n in H.nodes}, "osmid")
    G = ox.project_graph(H, to_crs="epsg:4326")
    return G


def remove_dead_end_edges(G: nx.MultiGraph, dead_end_tol: float = 50) -> nx.MultiGraph:
    dead_end_nodes = set()
    for n, n_degree in G.out_degree():
        if n_degree == 1:
            dead_end_nodes.add(n)

    dead_end_edges = set()
    for n in dead_end_nodes:
        u_out, v_out, l_out = list(G.out_edges(n, data="length"))[0]
        if l_out <= dead_end_tol:
            dead_end_edges.add((u_out, v_out, 0))
            dead_end_edges.add((v_out, u_out, 0))

            if G.out_degree(v_out) == 2:
                dead_end_length, dead_end_path = nx.single_source_dijkstra(G, source=v_out,
                                                                           cutoff=dead_end_tol - l_out, weight="length")
                target = max(dead_end_length, key=dead_end_length.get)
                u = v_out
                for v in dead_end_path[target][1:]:
                    if G.out_degree(v) == 2:
                        if u != v:
                            dead_end_edges.add((u, v, 0))
                            dead_end_edges.add((v, u, 0))
                            u = v
                    else:
                        break

    G.remove_edges_from(dead_end_edges)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(dead_end_edges)} dead-end edges.")
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
        ex_inf_type = ["track", "opposite_track", "separate"]
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

        if big_road:
            if not d["traffic_signals"]:
                d["intersection_size"] = "large"
            else:
                d["intersection_size"] = "medium"
        elif mid_road and not d["traffic_signals"]:
            d["intersection_size"] = "medium"
        else:
            d["intersection_size"] = "small"

        if G.out_degree(n) <= 2:
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
        car_only_street_types: list[str] | None = None,
        remove_car_only_edges: bool  = True,
        ensure_highway_diff: bool = False,
        consolidate: bool = False,
        intersection_tol: float = 35,
        remove_dead_ends: bool = False,
        dead_end_tol: float = 50,
        ex_inf: list[str] | None = None,
        elevation_raster_file: str | None = None,
) -> nx.MultiDiGraph:
    """Prepares the downloaded map. Removes all defined car-only edges. Turns it to undirected, removes all isolated nodes using
    networkxs isolates() function and reduces the graph to the greatest connected
    component using osmnxs get_largest_component().

    Parameters
    ----------
    G : nx.MultiGraph | nx.MultiDiGraph
        Graph to clean.
    car_only_street_types : list[str] | None
        List of car-only street types. (Default value = ["motorway", "trunk"])
    remove_car_only_edges : list[str] | None
        Remove car-only edges. (Default value = True)
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
    ex_inf : list[str] or None
        Which cycleways should be accepted as ex inf (Default value = None)
    elevation_raster_file : str | None = None
        Raster file with elevation data.

    Returns
    -------
    output : nx.MultiDiGraph

    """
    if ex_inf is None:
        ex_inf = ["track", "opposite_track", "separate"]
    if car_only_street_types is None:
        car_only_street_types = ["motorway", "motorway_link", "trunk", "trunk_link"]

    for u, v, k, d in G.edges(keys=True, data=True):
        d["highway"] = get_street_type(G, (u, v, k), motorway=car_only_street_types)

    # Check where bike paths/lanes already exist
    print("Check for existing infrastructure.")
    for u, v, k, d in G.edges(keys=True, data=True):
        if check_existing_infrastructure(d.get("cycleway", "missing"), ex_inf):
            d["ex_inf"] = True
        elif check_existing_infrastructure(d.get("cycleway:right", "missing"), ex_inf):
            d["ex_inf"] = True
        elif check_existing_infrastructure(d.get("cycleway:left", "missing"), ex_inf):
            d["ex_inf"] = True
        elif d.get("bicycle", "missing") == "use_sidepath":
            d["ex_inf"] = True
        elif d.get("sidewalk:both:bicycle", "missing") == "yes":
            d["ex_inf"] = True
        elif d.get("sidewalk:right:bicycle", "missing") == "yes":
            d["ex_inf"] = True
        elif d.get("sidewalk:left:bicycle", "missing") == "yes":
            d["ex_inf"] = True
        elif d.get("sidewalk:left:bicycle", "missing") == "designated":
            d["ex_inf"] = True
        elif d.get("cycleway:right:bicycle", "missing") == "designated":
            d["ex_inf"] = True
        elif d.get("cycleway:left:bicycle", "missing") == "designated":
            d["ex_inf"] = True
        elif d["highway"] == "bike_path":
            d["ex_inf"] = True
        else:
            d["ex_inf"] = False

    print(f"Starting graph cleaning with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    # Remove car-only edges
    if remove_car_only_edges:
        car_only_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data="highway") if d == "motorway"]
        G.remove_edges_from(car_only_edges)
        print(f"Removed {len(car_only_edges)} car only edges.")

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes.")

    traffic_signal_nodes = get_traffic_signal_nodes(G)

    # Simplify graph, remove irrelevant nodes
    if ensure_highway_diff:
        print("Simplifying graph, keeping highway differentiation.")
        G = ox.simplify_graph(G, edge_attrs_differ=["highway"])
    else:
        print("Simplifying graph, ignoring highway differentiation.")
        G = ox.simplify_graph(G)
    print(f"Simplified graph to {len(G.nodes())} nodes and {len(G.edges())} edges.")

    G = add_elevation_data(G, raster_file=elevation_raster_file)

    if consolidate:
        G = consolidate_nodes(G, intersection_tol)

    for n, d in G.nodes(data=True):
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

    G = nx.MultiDiGraph(ox.convert.to_undirected(G))
    G = nx.MultiDiGraph(ox.convert.to_digraph(G))

    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print(f"Removed {len(self_loops)} self loops.")
    
    if remove_dead_ends:
        G = remove_dead_end_edges(G, dead_end_tol=dead_end_tol)

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

    print(f"Final with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    return G


def download_map_by_bbox(
    bbox: tuple[float, float, float, float],
    network_type: str = "drive",
    car_only_street_types : list[str] | None = None,
    ensure_highway_diff: bool = False,
    consolidate: bool = False,
    intersection_tol: float = 35,
    remove_dead_ends: bool = False,
    dead_end_tol: float = 50,
    truncate_by_edge: bool = False,
    params: dict | None = None,
    paths: dict | None = None,
) -> nx.MultiDiGraph:
    """Downloads a drive graph from osm given by the bbox and cleans it for usage.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Boundary box of the map [east, south, west, north].
    network_type : str
        Network type for download, see ox.graph_from_bbox (Default value = "drive")
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
    truncate_by_edge : bool
        if True, retain node if it’s outside bounding box but at least one of node’s
        neighbors are within bounding box (Default value = False)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict | None
        Dict with the paths for data, plots etc.

    Returns
    -------
    output : nx.MultiDiGraph

    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()

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
    ox.settings.bidirectional_network_types = ox.settings.bidirectional_network_types + [network_type]
    G = ox.graph_from_bbox(
        bbox,
        network_type=network_type,
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, car_only_street_types=car_only_street_types, ensure_highway_diff=ensure_highway_diff, consolidate=consolidate, intersection_tol=intersection_tol, remove_dead_ends=remove_dead_ends, dead_end_tol=dead_end_tol, ex_inf=params["ex inf"], elevation_raster_file=paths["elevation_raster_file"]
    )

    return G


def download_map_by_name(
    city: str,
    nominatim_result: int = 1,
    network_type: str = "drive",
    car_only_street_types : list[str] | None = None,
    ensure_highway_diff: bool = False,
    consolidate: bool = False,
    intersection_tol: float = 35,
    remove_dead_ends: bool = False,
    dead_end_tol: float = 50,
    truncate_by_edge: bool = False,
    params: dict | None = None,
    paths: dict | None = None,
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
    car_only_street_types : list[str] | None
        List of generally car-only street types to keep. (Default value = ["motorway", "trunk"])
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
    truncate_by_edge : bool
        if True, retain node if it’s outside the area but at least one of node’s
        neighbors are within the area (Default value = False)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict | None
        Dict with the paths for data, plots etc.

    Returns
    -------
    output : nx.MultiDiGraph

    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()

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
    ox.settings.bidirectional_network_types = ox.settings.bidirectional_network_types + [network_type]
    G = ox.graph_from_place(
        city,
        which_result=nominatim_result,
        network_type=network_type,
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, car_only_street_types=car_only_street_types, ensure_highway_diff=ensure_highway_diff, consolidate=consolidate,
        intersection_tol=intersection_tol, remove_dead_ends=remove_dead_ends, dead_end_tol=dead_end_tol,
        ex_inf=params["ex inf"], elevation_raster_file=paths["elevation_raster_file"]
    )

    return G


def download_map_by_polygon(
    polygon,
    network_type: str = "drive",
    car_only_street_types : list[str] | None = None,
    ensure_highway_diff: bool = False,
    consolidate: bool = False,
    intersection_tol: float = 35,
    remove_dead_ends: bool = False,
    dead_end_tol: float = 50,
    truncate_by_edge: bool = False,
    params: dict | None = None,
    paths: dict | None = None,
) -> nx.MultiDiGraph:
    """Downloads a drive graph from osm given by the polygon and cleans it for
    usage.

    Parameters
    ----------
    polygon : Polygon
        Polygon of the area.
    network_type : str
        Network type for download, see ox.graph_from_bbox (Default value = "drive")
    car_only_street_types : list[str] | None
        List of generally car-only street types to keep. (Default value = ["motorway", "trunk"])
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
    truncate_by_edge : bool
        if True, retain node if it’s outside bounding box but at least one of node’s
        neighbors are within bounding box (Default value = False)
    params : dict | None
        Params for data loading etc., check 'setup_params.py' in the 'scripts' folder.
        (Default value = None)
    paths : dict | None
        Dict with the paths for data, plots etc.

    Returns
    -------
    output :  nx.MultiDiGraph

    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()

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
    ox.settings.bidirectional_network_types = ox.settings.bidirectional_network_types + [network_type]
    G = ox.graph_from_polygon(
        polygon,
        network_type=network_type,
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, car_only_street_types=car_only_street_types, ensure_highway_diff=ensure_highway_diff, consolidate=consolidate,
        intersection_tol=intersection_tol, remove_dead_ends=remove_dead_ends, dead_end_tol=dead_end_tol,
        ex_inf=params["ex inf"], elevation_raster_file=paths["elevation_raster_file"]
    )

    return G


def generate_all_to_all_demand(g: nx.MultiGraph | nx.MultiDiGraph, n_ij: dict | None = None):
    """
    Generate demand between all nodes of graph g with `n_ij` number of travellers.
    Parameters
    ----------
    g: nx.MultiGraph | nx.MultiDiGraph
    n_ij: dict | None

    Returns
    -------

    """
    if n_ij is None:
        n_ij = {1: 1}
    nodes = [n for n, d in g.nodes(data=True) if not d["bike_highway"]]
    nodes = list(it.product(nodes, repeat=2))
    nodes = [od for od in nodes if od[0] != od[1]]

    demand = {(od[0], od[1]): n_ij for od in nodes}

    return demand


def noisy_demand(
    demand_dict: dict,
    delta: float,
    nodes_noise: dict,
    rng: np.random.Generator = None,
    ):
    """

    Parameters
    ----------
    demand_dict
    delta
    nodes_noise
    rng

    Returns
    -------

    """
    if rng is None:
        rng = np.random.default_rng()

    if not set(type(k) for k in demand_dict.keys()) == {tuple}:
        noisy_demand_dict = {literal_eval(od): deepcopy(trips) for od, trips in demand_dict.items()}
    else:
        noisy_demand_dict = {od: deepcopy(trips) for od, trips in demand_dict.items()}

    if nodes_noise is None:
        nodes = set()
        for od in noisy_demand_dict.keys():
            nodes.add(od[0])
            nodes.add(od[1])
        nodes_noise = {node: rng.uniform(1 - delta, 1 + delta) for node in nodes}

    for od, noisy_trips in noisy_demand_dict.items():
        for c_type in noisy_trips.keys():
            demand_old = demand_dict[f"{od}"][c_type]
            noise_factor = (nodes_noise[od[0]] + nodes_noise[od[1]]) / 2
            noisy_demand_dict[od][c_type] = demand_old * noise_factor
    return noisy_demand_dict, nodes_noise


def noisy_penalty(
    penalty_dict: dict,
    delta: float,
    max_penalty: float = 1.0,
    min_penalty: float = 0.0,
    rng: np.random.Generator = None,
    ):
    if rng is None:
        rng = np.random.default_rng()

    high = 1 + delta
    low = 1 - delta

    noisy_penalty_dict = deepcopy(penalty_dict)
    noisy_penalty_dict = dict(sorted(noisy_penalty_dict.items(), key=lambda x: x[1][0]))

    for k, old_penalties in noisy_penalty_dict.items():
        for idx, old_penalty in enumerate(old_penalties):
            if len(old_penalties) == 1:
                old_penalty_lower = -np.inf
                old_penalty_higher = np.inf
            else:
                if idx == 0:
                    old_penalty_lower = -np.inf
                    old_penalty_higher = old_penalties[idx + 1]
                elif idx == len(old_penalties) - 1:
                    old_penalty_lower = old_penalties[idx - 1]
                    old_penalty_higher = np.inf
                else:
                    old_penalty_lower = old_penalties[idx - 1]
                    old_penalty_higher = old_penalties[idx + 1]

            sorted_penalties_tier = [v[idx] for v in noisy_penalty_dict.values()]
            sorted_penalties_tier.sort()

            old_penalty_index_tier = bisect.bisect(sorted_penalties_tier, old_penalty)

            if len(sorted_penalties_tier) == 1:
                old_penalty_lower_tier = -np.inf
                old_penalty_higher_tier = np.inf
            elif old_penalty_index_tier <= 1:
                old_penalty_lower_tier = -np.inf
                old_penalty_higher_tier = sorted_penalties_tier[old_penalty_index_tier]
            elif old_penalty_index_tier == len(sorted_penalties_tier):
                old_penalty_lower_tier = sorted_penalties_tier[old_penalty_index_tier - 2]
                old_penalty_higher_tier = np.inf
            else:
                old_penalty_lower_tier = sorted_penalties_tier[old_penalty_index_tier - 2]
                old_penalty_higher_tier = sorted_penalties_tier[old_penalty_index_tier]

            new_penalty = old_penalty * rng.uniform(low, high)

            while new_penalty < old_penalty_lower_tier or new_penalty > old_penalty_higher_tier or new_penalty < old_penalty_lower or new_penalty > old_penalty_higher:
                new_penalty = old_penalty * rng.uniform(low, high)

            new_penalty = min(max_penalty, new_penalty)
            new_penalty = max(min_penalty, new_penalty)
            noisy_penalty_dict[k][idx] = new_penalty

    return noisy_penalty_dict


def get_k_neighbours(g, n, k=1):
    if k == 1:
        return list(nx.neighbors(g, n))
    else:
        g_k_1 = nx.ego_graph(g, n, radius=k-1)
        g_k = nx.ego_graph(g, n, radius=k)
        return [n for n in g_k.nodes() if n not in g_k_1.nodes()]


def bearing(x1: float | list[float], y1: float | list[float], x2: float | list[float], y2: float | list[float]):
    """
    Calculate the compass bearing(s) between pairs of x-y points.

    Parameters
    ----------
    x1 : float | list[float]
        x coordinate of the first point(s).
    y1 : float | list[float]
        y coordinate of the first point(s).
    x2 : float | list[float]
        x coordinate of the second point(s).
    y2 : float | list[float]
        y coordinate of the second point(s).

    Returns
    -------
    output : numpy.ndarray
        Bearing(s)
    """
    theta = np.arctan2(x2 - x1, y1 - y2)
    return np.degrees(theta)


def add_edge_bearings(g: nx.MultiDiGraph, precision: int = 1):
    """
    Simplified copy of osmnx "add_edge_bearings".
    Add compass `bearing` attributes to all graph edges.

    Parameters
    ----------
    g : nx.MultiDiGraph
        Graph
    precision : int
        Precision of the bearing

    Returns
    -------
    output : nx.MultiDiGraph
        Graph g with edge bearings in edge attribute "bearing".
    """
    # extract edge IDs and corresponding coordinates from their nodes
    uvk = [(u, v, k) for u, v, k in g.edges if u != v]
    x = g.nodes(data="x")
    y = g.nodes(data="y")
    coords = np.array([(y[u], x[u], y[v], x[v]) for u, v, k in uvk])

    # calculate bearings then set as edge attributes
    bearings = bearing(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])
    values = zip(uvk, bearings.round(precision))
    nx.set_edge_attributes(g, dict(values), name="bearing")

    return g


def bearing_walk(g, start, bear, steps):
    queue = [start]
    step = 0
    while len(queue) != 0:
        n = queue.pop(0)
        for u, v, d in g.out_edges(n, data=True):
            if d["bearing"] == bear and not d["bike_highway"]:
                queue.append(v)
                step += 1
            if step == steps:
                return v
    return start


def n_dist_bearing(g, origin, dist):
    g_bearing = nx.MultiDiGraph(g)
    add_edge_bearings(g_bearing)
    visited = set()
    queue = [origin]
    nodes = {origin}
    while len(queue) != 0:
        n = queue.pop(0)
        if n not in visited:
            visited.add(n)
            new_nodes = set()
            for out_bearing in [d for u, v, d in g_bearing.out_edges(n, data="bearing")]:
                new_node = bearing_walk(g_bearing, n, out_bearing, dist)
                if new_node not in visited and not g_bearing.nodes[new_node]["bike_highway"]:
                    new_nodes.add(new_node)
            nodes = nodes.union(new_nodes)
            queue += [v for v in new_nodes]
    return nodes


def add_osmnx_data(g: nx.MultiDiGraph):
    """
    Adds osmnx-like data to the graph, altitude data is drawn from the
    surface input.

    Parameters
    ----------
    g : nx.MultiDiGraph
        Graph

    Returns
    -------
    output : networkx.MultiDiGraph
        Graph with relevant osmnx-like data.
    """
    g.graph["crs"] = "epsg:4326"

    nx.set_node_attributes(g, {v: x for v, x in g.nodes(data="x")}, "lon")
    nx.set_node_attributes(g, {v: y for v, y in g.nodes(data="y")}, "lat")

    traffic_lights = list(g.nodes())
    nx.set_node_attributes(g, {n: "medium" if n in traffic_lights else "large" for n in g.nodes()}, "intersection_size")
    for n, d in g.nodes(data=True):
        if d["bike_highway"]:
            d["intersection_size"] = "small"

    add_edge_bearings(g)

    nx.set_node_attributes(g, {n: 0.0 for n in g.nodes()}, "elevation")
    ox.add_edge_grades(g, add_absolute=True)

    nx.set_edge_attributes(g, {(u, v, k): "primary" for u, v, k in g.edges(keys=True)}, "highway")
    nx.set_edge_attributes(g, {(u, v, k): 50 for u, v, k in g.edges(keys=True)}, "maxspeed")
    nx.set_edge_attributes(g, {(u, v, k): LineString([(g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])]) for (u, v, k) in g.edges(keys=True)}, "geometry")
    for u, v, d in g.edges(data=True):
        if "ramp" in d.keys():
            d["ex_inf"] = d["ramp"]
        else:
            d["ex_inf"] = False

    set_turn_penalty(g)

    return g


def triangular_bike_highways(g: nx.MultiDiGraph, central_node=None, lengths=None, epsilon=1e-06):
    """

    Parameters
    ----------
    g
    central_node
    lengths
    epsilon

    Returns
    -------

    """
    if central_node is None:
        bc = nx.betweenness_centrality(g)
        central_node = max(bc, key=bc.get)
    if lengths is None:
        lengths = [2]

    for length in lengths:
        new_ramp_nodes = n_dist_bearing(g, central_node, length)
        if len(new_ramp_nodes) > 1:
            highway_nodes_level = {d["rampnode"]: n for n, d in g.nodes(data=True) if
                                   "rampnode" in d.keys() and "level" in d.keys() if d["level"] == length - 1}
            sp_len = dict()
            i = len(g.nodes)
            for n in new_ramp_nodes:
                if n not in highway_nodes_level.keys():
                    highway_nodes_level[n] = i
                    g.add_node(i, **g.nodes[n])
                    g.nodes[i]["bike_highway"] = True
                    g.nodes[i]["rampnode"] = n
                    g.nodes[i]["level"] = length - 1
                    i += 1
                sp_len[n] = {m: nx.shortest_path_length(g, source=n, target=m, weight="length") for m in new_ramp_nodes}

            for u, sp_len_u in sp_len.items():
                for v, sp_len_u_v in sp_len_u.items():
                    if sp_len_u_v == length:
                        g.add_edge(highway_nodes_level[u], highway_nodes_level[v], length=length, bike_highway=True, blocked=False)

    for ramp_node, highway_node in [(d["rampnode"], n) for n, d in g.nodes(data=True) if "rampnode" in d.keys()]:
        g.add_edge(ramp_node, highway_node, length=epsilon, bike_highway=True, ramp=True, blocked=True)