"""
This module includes all necessary helper functions for the data preparation
and handling.
"""
import ast
import h5py
import json
import geog
import srtm
import warnings
import itertools as it
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
from itertools import product
from collections import Counter
from os.path import join
from scipy import interpolate
from operator import itemgetter
from math import cos, asin, sqrt, pi
from shapely.geometry import Point, Polygon
from .setup_helper import create_default_paths, create_default_params


def read_csv(path, delim=","):
    """
    Reads the csv given by path. Delimiter of csv can be chosen by delim.
    All column headers ar converted to lower case.
    :param path: path to load csv from
    :type path: str
    :param delim: delimiter of csv
    :type delim: str
    :return: data frame
    :rtype: pandas DataFrame
    """
    df = pd.read_csv(path, delimiter=delim)
    df.columns = map(str.lower, df.columns)
    return df


def write_csv(df, path):
    """
    Writes given data frame to csv.
    :param df: data frame
    :type df: pandas DataFrame
    :param path: path to save
    :type path: str
    :return: None
    """
    df.to_csv(path, index=False)


def distance(lat1, lon1, lat2, lon2):
    """
    Calcuate the distance between two lat/long points in meters.
    :param lat1: Latitude of point 1
    :type lat1: float
    :param lon1: Longitude of pint 1
    :type lon1: float
    :param lat2: Latitude of point 2
    :type lat2: float
    :param lon2: Longitude of pint 2
    :type lon2: float
    :return: Distance in meters
    :rtype: float
    """
    p = pi / 180
    a = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(a))


def get_circle_from_point(point, radius, n_points=20):
    """
    Returns a circle around a long/lat point with given radius.
    :param point: point (lon, lat)
    :type point: tuple
    :param radius: radius of the circle
    :type radius: float
    :param n_points: number of sides of the polygon
    :type n_points: int
    :return: circle (polygon)
    :rtype: shapely Polygon
    """
    p = Point(point)
    angles = np.linspace(0, 360, n_points)
    polygon = geog.propagate(p, angles, radius)
    return Polygon(polygon)


def get_nodes_in_radius(G, nodes, radius, nbr_nn=None):
    """

    :param G:
    :param nodes:
    :param radius:
    :param nbr_nn:
    :return:
    """
    orig_nodes = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in nodes}
    all_nodes = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes()}

    nodes_in_r = {}
    for n, coords in orig_nodes.items():
        circle = get_circle_from_point(coords, radius=radius)
        n_in_r = [i for i, c in all_nodes.items() if circle.contains(Point(c))]
        if nbr_nn is not None:
            n_in_r = np.random.choice(n_in_r, min(len(n_in_r), nbr_nn), replace=False)
        nodes_in_r[n] = n_in_r

    return nodes_in_r


def get_street_length(G, edge):
    """
    Returns the street length of the edge.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx graph.
    :type edge: tuple of integers
    :return: Street type·
    :rtype: str
    """
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        return G[edge[0]][edge[1]][0]["length"]
    else:
        return G[edge[0]][edge[1]]["length"]


def get_street_type(G, edge, motorway=None):
    """
    Returns the street type of the edge. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx graph.
    :type edge: tuple of integers
    :return: Street type·
    :rtype: str
    """
    if motorway is None:
        motorway = ["motorway", "motorway_link"]
    if "trunk" not in motorway:
        primary = ["primary", "primary_link", "trunk", "trunk_link"]
    else:
        primary = ["primary", "primary_link"]

    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        street_type = G[edge[0]][edge[1]][0]["highway"]
    else:
        street_type = G[edge[0]][edge[1]]["highway"]
    if isinstance(street_type, list):
        street_type = street_type[0]
    elif isinstance(street_type, str):
        street_type = street_type
    else:
        print(f"Street type unusable: {street_type}")
    if street_type in motorway:
        return "motorway"
    if street_type in primary:
        return "primary"
    elif street_type in ["secondary", "secondary_link"]:
        return "secondary"
    elif street_type in ["tertiary", "tertiary_link"]:
        return "tertiary"
    else:
        return "residential"


def broaden_demand(G, trips, radius=100, nbr_nn=None):
    trips = {(o, d): n for (o, d), n in trips.items() if o != d}
    stations = set()
    for k, v in trips.items():
        stations.add(k[0])
        stations.add(k[1])

    print(f"Number of old trips: {sum([v for v in trips.values()])}")
    print(f"Number of old stations: {len(stations)}")

    new_stations = get_nodes_in_radius(G, stations, radius, nbr_nn=nbr_nn)

    new_trips = Counter({})
    for (o, d), n in trips.items():
        new_o = new_stations[o]
        new_d = new_stations[d]
        new_t = {
            (n_o, n_d): n // (len(new_o) * len(new_d))
            for (n_o, n_d) in it.product(new_o, new_d)
        }
        idx = [
            list(new_t.keys())[i]
            for i in np.random.choice(
                len(list(new_t.keys())),
                n % (len(new_o) * len(new_d)),
                replace=False,
            )
        ]
        for i in idx:
            new_t[i] += 1
        new_trips = new_trips + Counter(new_t)

    new_stations = set()
    for k, v in new_trips.items():
        new_stations.add(k[0])
        new_stations.add(k[1])
    print(f"Number of new trips: {sum([v for v in new_trips.values()])}")
    print(f"Number of new stations: {len(new_stations)}")
    return dict(new_trips)


def get_lat_long_trips(path_to_trips, polygon=None, delim=","):
    """
    Returns five lists. The first stores the number of cyclists on this trip,
    the second the start latitude, the third the start longitude,
    the fourth the end latitude, the fifth the end longitude.
    An index corresponds to the same trip in each list.
    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :param delim: Delimiter of the trips csv.
    :type delim: str
    :return: number of trips, start lat, start long, end lat, end long
    :rtype: list
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


def get_bbox_of_trips(path_to_trips, polygon=None, delim=","):
    """
    Returns the bbox of the trips given by path_to_trips. If only trips inside
    a polygon should be considered, you can pass it to the polygon param.
    :param path_to_trips: path to the compacted trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
     pass it here.
    :type polygon: Shapely Polygon
    :param delim: Delimiter of the trips csv.
    :type delim: str
    :return: list of bbox [north, south, east, west]
    :rtype: list
    """
    trips_used, start_lat, start_long, end_lat, end_long = get_lat_long_trips(
        path_to_trips, polygon, delim=delim
    )
    north = max(start_lat + end_lat) + 0.005
    south = min(start_lat + end_lat) - 0.005
    east = max(start_long + end_long) + 0.01
    west = min(start_long + end_long) - 0.01
    return [north, south, east, west]


def load_trips(G, path_to_trips, polygon=None, delim=","):
    """
    Loads the trips and maps lat/long of start and end station to nodes in
    graph G. For this the ox.distance.nearest_nodes function of osmnx is used.
    :param G: graph used for lat/long to node mapping
    :param path_to_trips: path to the trips csv.
    :type path_to_trips: str
    :param polygon: If only trips inside a polygon should be considered,
    pass it here.
    :type polygon: Shapely Polygon
    :param delim: Delimiter of the trips csv.
    :type delim: str
    :return: dict with trip info and set of stations used.
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

    trip_nbrs_rexcl = {k: v for k, v in trip_nbrs.items() if not k[0] == k[1]}
    print(
        f"Number of Stations: {len(stations)}, "
        f"Number of trips: {sum(trip_nbrs_rexcl.values())} "
        f"(rt incl: {sum(trip_nbrs.values())}), "
        f"Unique trips: {len(trip_nbrs_rexcl.keys())} "
        f"(rt incl {len(trip_nbrs.keys())})"
    )
    return trip_nbrs, stations


def trip_cyclist_type_split(trips, cyclist_split=None):
    """

    :param trips:
    :param cyclist_split:
    :return:
    """
    if cyclist_split is None:
        cyclist_split = {1: 1}
    return {
        trip_od: {cyclist: trip_nbr * split for cyclist, split in cyclist_split.items()}
        for trip_od, trip_nbr in trips.items()
    }


def get_polygon_from_json(path_to_json):
    """
    Reads json at path. json can be created at http://geojson.io/.
    :param path_to_json: file path to json.
    :type path_to_json: str
    :return: Polygon given by json
    :rtype: Shapely polygon
    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    coordinates = data["features"][0]["geometry"]["coordinates"][0]
    coordinates = [(item[0], item[1]) for item in coordinates]
    polygon = Polygon(coordinates)
    return polygon


def get_polygons_from_json(path_to_json):
    """
    Reads json at path. json can be created at http://geojson.io/.
    :param path_to_json: file path to json.
    :type path_to_json: str
    :return: Polygon given by json
    :rtype: Shapely polygon
    """
    with open(path_to_json) as j_file:
        data = json.load(j_file)
    polygons = []
    for d in data["features"]:
        coordinates = d["geometry"]["coordinates"][0]
        coordinates = [(item[0], item[1]) for item in coordinates]
        polygons.append(Polygon(coordinates))
    return polygons


def get_polygon_from_bbox(bbox):
    """
    Returns the Polygon resembled by the given bbox.
    :param bbox: bbox [north, south, east, west]
    :type bbox: list
    :return: Polygon of bbox
    :rtype: Shapely Polygon
    """
    north, south, east, west = bbox
    corners = [(east, north), (west, north), (west, south), (east, south)]
    polygon = Polygon(corners)
    return polygon


def get_bbox_from_polygon(polygon):
    """
    Returns bbox from given polygon.
    :param polygon: Polygon
    :type polygon: Shapely Polygon
    :return: bbox [north, south, east, west]
    :rtype: list
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

    return [north, south, east, west]


def drop_invalid_values(csv, column, values, save=False, save_path="", delim=","):
    """
    Drops all rows if they have the given invalid value in the given column.
    :param csv: Path to csv.
    :type csv: str
    :param column: Column naem which should be checked vor invalid values.
    :type column: str
    :param values: List of the invalid values in the column.
    :type values: list
    :param save: Set true if df should be saved as csv after dropping.
    :type save: bool
    :param save_path: Path where it should e saved.
    :type save_path: str
    :param delim: Delimiter of the original csv.
    :type delim: str
    :return: DataFrame without the invalid values.
    :rtype: pandas DataFrame
    """
    df = read_csv(csv, delim)
    for value in values:
        drop_ind = df[df[column] == value].index
        df.drop(drop_ind, inplace=True)
    if save:
        write_csv(df, save_path)
    return df


def consolidate_nodes(G, tol):
    """
    Consolidates intersections of graph g with given tolerance in meters.
    :param G: Graph to consolidate intersections in
    :type G: networkx (Multi)(Di)Graph
    :param tol: Tolerance for consolidation in meters
    :type tol: float or int
    :return: Graph with consolidated intersections
    :rtype: same as param G
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


def get_traffic_signal_nodes(G, tol=10):
    """
    Returns the OSMids of the intersections with traffic signals.
    :param G: Graph to consolidate intersections in
    :type G: networkx (Multi)(Di)Graph
    :param tol: Tolerance for consolidation in meters
    :type tol: float or int
    :return: Set of OSMids of intersections with traffic signals.
    :rtype: set
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
            d["osmid_original"] = ast.literal_eval(d["osmid_original"])
        else:
            d["osmid_original"] = [int(d["osmid_original"])]
        for o_id in d["osmid_original"]:
            if (
                "highway" in G.nodes[o_id].keys()
                and G.nodes[o_id]["highway"] == "traffic_signals"
            ):
                traffic_signal_list += d["osmid_original"]

    return set(traffic_signal_list)


def check_existing_infrastructure(exinf, ex_inf_type=None):
    """
    Check if the existing bike path/lane infrastructure meets the required
    type defined in ex_inf_type or not. Check OSM wiki for possible values.
    https://wiki.openstreetmap.org/wiki/Key:cycleway
    :param exinf: Existing bike lane/path
    :type exinf: str or list of str
    :param ex_inf_type: List of infrastructure types which are accepted.
    :type ex_inf_type: List of str
    :return: Returns True if existing infrastructure type meets the
    requirements or False if not.
    :rtype: bool
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


def get_turn_angle(edge_in, edge_out, G):
    """
    Returns the turn angle between out- and in-edge in Graph G.
    The angle is calculated counter-clockwise (driving on the right).
    :param edge_in: Incoming edge.
    :type edge_in: tuple
    :param edge_out: Outgoing edge.
    :type edge_out: tuple
    :return: angle between out- and in-edge in degrees.
    :rtype: float
    """
    if isinstance(G, nx.MultiDiGraph) or isinstance(G, nx.MultiGraph):
        e_in = G[edge_in[0]][edge_in[1]][0]
        e_out = G[edge_out[0]][edge_out[1]][0]
    else:
        e_in = G[edge_in[0]][edge_in[1]]
        e_out = G[edge_out[0]][edge_out[1]]

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


def get_turn_order(edge_out, G):
    """
    Returns the incoming edges to edge_outs star node.
    The edges are ordered by turn angle from smallest to largest.
    The angles are calculated counter-clockwise (driving on the right).
    :param edge_out: Outgoing edge.
    :type edge_out: tuple
    :param G: Graph to consolidate intersections in
    :type G: networkx (Multi)(Di)Graph
    :return: List of ingoing edges ordered by turnangle.
    :rtype: list
    """
    n = edge_out[0]
    in_neighbors = list(G.predecessors(n))
    in_edges = [(n_in, n) for n_in in in_neighbors]
    turns = {edge_in: get_turn_angle(edge_in, edge_out, G) for edge_in in in_edges}

    return [k for k, v in sorted(turns.items(), key=lambda item: item[1])]


def check_intersection_size(G):
    """
    Sets the size of the intersections in graph G.
    :param G: Graph to consolidate intersections in
    :type G: networkx (Multi)(Di)Graph
    :return: G
    :rtype: same as input type
    """
    for n, d in G.nodes(data=True):
        big_road = False
        mid_road = False
        for i in list(set(nx.all_neighbors(G, n))):
            s_type = get_street_type(G, (i, n))
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


def set_turn_penalty(G):
    """
    Sets the turn penalty of the intersections in graph G.
    :param G: Graph to consolidate intersections in
    :type G: networkx (Multi)(Di)Graph
    :return: G
    :rtype: same as input type
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


def prepare_downloaded_map(G, trunk=False, consolidate=False, tol=35, ex_inf=None):
    """
    Prepares the downloaded map. Removes all motorway edges and if
    trunk=False also all trunk edges. Turns it to undirected, removes all
    isolated nodes using networkxs isolates() function and reduces the graph to
    the greatest connected component using osmnxs get_largest_component().
    :param G: Graph to clean.
    :type G: networkx graph
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :param ex_inf: Which cycleways should be accepted as ex inf
    :type ex_inf: list of strings
    :return: Cleaned graph
    :rtype: networkx graph.
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
        e for e in G.edges() if get_street_type(G, e, motorway=s_t) in s_t
    ]
    G.remove_edges_from(edges_to_remove)
    print(f"Removed {len(edges_to_remove)} car only edges.")

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes.")
    G = ox.utils_graph.get_largest_component(G)
    print("Reduce to largest connected component")

    traffic_signal_nodes = get_traffic_signal_nodes(G)

    # Simplify graph, remove irrelevant nodes
    G = ox.simplify_graph(G)

    if consolidate:
        G = consolidate_nodes(G, tol)

    srtm_data = srtm.Srtm1HeightMapCollection()
    for n, d in G.nodes(data=True):
        lat = d["y"]
        lng = d["x"]
        try:
            d["elevation"] = srtm_data.get_altitude(latitude=lat, longitude=lng)
        except srtm.exceptions.NoHeightMapDataException:
            warnings.warn(
                f"SRTM directory '{srtm_data.hgt_dir}' is missing files. Setting height 0 for all nodes."
            )
            d["elevation"] = 0
        if consolidate:
            if isinstance(d["osmid_original"], str):
                node_set = set(ast.literal_eval(d["osmid_original"]))
            else:
                node_set = {d["osmid_original"]}
            if len(node_set.intersection(traffic_signal_nodes)) != 0:
                d["traffic_signals"] = True
            else:
                d["traffic_signals"] = False
        else:
            if "highway" in d.keys() and d["highway"] == "traffic_signals":
                d["traffic_signals"] = True
            else:
                d["traffic_signals"] = False

    G = nx.MultiDiGraph(ox.get_digraph(G))
    G = nx.MultiDiGraph(ox.get_undirected(G))

    # Calculate slope in percent
    print("Adding street gradients.")
    G = ox.add_edge_grades(G, add_absolute=True)

    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print(f"Removed {len(self_loops)} self loops.")

    # Set intersection penalties
    print("Setting intersection penalties.")
    G = ox.add_edge_bearings(G)
    G = check_intersection_size(G)
    G = set_turn_penalty(G)

    for u, v, d in G.edges(data=True):
        if "ex_inf" in d.keys():
            if isinstance(d["ex_inf"], list):
                if True in d["ex_inf"]:
                    d["ex_inf"] = True
                else:
                    d["ex_inf"] = False
        else:
            d["ex_inf"] = False
        d["highway"] = get_street_type(G, (u, v))
        d["cost"] = d["length"]

    G = nx.MultiDiGraph(ox.get_digraph(G))
    G = nx.MultiDiGraph(ox.get_undirected(G))

    return G


def download_map_by_bbox(
    bbox,
    trunk=False,
    consolidate=False,
    tol=35,
    truncate_by_edge=False,
    params=None,
):
    """
    Downloads a drive graph from osm given by the bbox and cleans it for usage.
    :param bbox: Boundary box of the map.
    :type bbox: list [north, south, east, west]
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :param truncate_by_edge: if True, retain node if it’s outside bounding box
    but at least one of node’s neighbors are within bounding box
    :type truncate_by_edge: bool
    :param params: Dictionary with parameters for plotting etc
    :type params: dict or None
    :return: Cleaned graph.
    :rtype: networkx graph
    """
    if params is None:
        params = create_default_params()

    print(
        f"Downloading map from bounding box. "
        f"Northern bound: {bbox[0]}, "
        f"southern bound: {bbox[1]}, "
        f"eastern bound: {bbox[2]}, "
        f"western bound: {bbox[3]}"
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
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        network_type="drive",
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, trunk, consolidate=consolidate, tol=tol, ex_inf=params["ex inf"]
    )

    return G


def download_map_by_name(
    city,
    nominatim_result=1,
    trunk=False,
    consolidate=False,
    tol=35,
    truncate_by_edge=False,
    params=None,
):
    """
    Downloads a drive graph from osm given by the name and geocode of the
    nominatim database and  cleans it for usage.
    :param city: Name of the place to donload.
    :type city: str
    :param nominatim_result: Which result of the nominatim database should
    be downloaded.
    :type nominatim_result: int
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :param truncate_by_edge: if True, retain node if it’s outside bounding box
    but at least one of node’s neighbors are within bounding box
    :type truncate_by_edge: bool
    :param params: Dictionary with parameters for plotting etc
    :type params: dict or None
    :return: Cleaned graph.
    :rtype: networkx graph
    """
    if params is None:
        params = create_default_params()

    print(
        f"Downloading map py place. Name of the place: {city}, "
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
        network_type="drive",
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, trunk, consolidate=consolidate, tol=tol, ex_inf=params["ex inf"]
    )

    return G


def download_map_by_polygon(
    polygon,
    trunk=False,
    consolidate=False,
    tol=35,
    truncate_by_edge=False,
    params=None,
):
    """
    Downloads a drive graph from osm given by the polygon and cleans it for
    usage.
    :param polygon: Polygon of the graph.
    :type polygon: shapely Polygon
    :param trunk: Decides if trunk should be kept or not. If you want to
    keep trunk in the graph, set to True.
    :type trunk: bool
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :param truncate_by_edge: if True, retain node if it’s outside bounding box
    but at least one of node’s neighbors are within bounding box
    :type truncate_by_edge: bool
    :param params: Dictionary with parameters for plotting etc
    :type params: dict or None
    :return: Cleaned graph.
    :rtype: networkx graph
    """
    if params is None:
        params = create_default_params()

    print("Downloading map py polygon.")
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
        network_type="drive",
        truncate_by_edge=truncate_by_edge,
        simplify=False,
    )

    G = prepare_downloaded_map(
        G, trunk, consolidate=consolidate, tol=tol, ex_inf=params["ex inf"]
    )

    return G


def save_demand(trips, save_folder, save_name):
    """ """
    with open(join(save_folder, f"{save_name}_demand.json"), "w") as fp:
        json.dump({str(k): v for k, v in trips.items()}, fp)


def save_graph(G, save_folder, save_name):
    """
    Saves graph to given path.
    :param G: Graph to save.
    :type G: networkx graph
    :param save_folder: Path to save folder.
    :type save_folder: str
    :param save_name: Name of the graphml file.
    :type save_name: str
    :return: none
    """
    edge_data = {
        str((u, v)): {
            "length": d["length"],
            "street_type": d["highway"],
            "ex_inf": d["ex_inf"],
            "turn_penalty": d["turn_penalty"],
            "slope": d["grade"],
            "cost": d["cost"],
            "bike_highway": False,
        }
        for u, v, d in G.edges(data=True)
    }
    node_data = {
        n: {
            "intersection_size": d["intersection_size"],
            "bike_highway": False,
        }
        for n, d in G.nodes(data=True)
    }
    with open(join(save_folder, f"{save_name}_graph.json"), "w") as fp:
        json.dump({"node_data": node_data, "edge_data": edge_data}, fp)
    ox.save_graphml(G, filepath=join(save_folder, f"{save_name}.graphml"))


def data_to_matrix(stations, trips):
    """
    Converts given od demand into origin-destination matrix.
    :param stations: Stations of the demand
    :type stations: list
    :param trips: Demand
    :type trips: dict
    :return: OD Matrix
    :rtype: pandas dataframe
    """
    df = pd.DataFrame(stations, columns=["station"])
    for station in stations:
        df[station] = [np.nan for x in range(len(stations))]
    df.set_index("station", inplace=True)
    for k, v in trips.items():
        if not k[0] == k[1]:
            df[k[0]][k[1]] = v
    return df


def matrix_to_graph(df, rename_columns=None, data=True):
    """
    Turns OD Matrix to graph.
    :param df: OD Matrix
    :type df: pandas dataframe
    :param rename_columns: If columns of the df should be renamed set
    appropriate dict here.
    :type rename_columns: dict
    :param data: If metadata of the demand (degree, indegree, outdegree,
    imbalance) should be returned or not.
    :type data: bool
    :return: Graph and (if wanted) meta data
    :rtype: networkx graph and list, list, list, list
    """
    if rename_columns is None:
        rename_columns = {"station": "source", "level_1": "target", 0: "trips"}
    df.values[[np.arange(len(df))] * 2] = np.nan
    df = df.stack().reset_index()
    df = df.rename(columns=rename_columns)
    g = nx.from_pandas_edgelist(df=df, edge_attr="trips", create_using=nx.MultiDiGraph)
    edge_list = list(g.edges())
    for u, v, d in g.edges(data="trips"):
        if (v, u) in edge_list:
            g[v][u][0]["total trips"] = d + g[v][u][0]["trips"]
            g[v][u][0]["imbalance"] = abs(d - g[v][u][0]["trips"]) / max(
                d, g[v][u][0]["trips"]
            )
        else:
            g[u][v][0]["total trips"] = d
            g[u][v][0]["imbalance"] = 1
    if data:
        indegree = [d for n, d in g.in_degree()]
        outdegree = [d for n, d in g.out_degree()]
        g = nx.Graph(g)
        degree = [d for n, d in nx.degree(g)]
        imbalance = [d for u, v, d in g.edges(data="imbalance")]
        for u, v, d in g.edges(data="total trips"):
            g[u][v]["trips"] = d
        return g, degree, indegree, outdegree, imbalance
    else:
        g = nx.Graph(g)
        for u, v, d in g.edges(data="total trips"):
            g[u][v]["trips"] = d
        return g


def sort_clustering(G):
    """
    Sorts nodes of G by clustering coefficient.
    :param G: Graph to sort
    :type G: networkx graph
    :return: List of nodes sorted by clustering coefficient.
    :rtype: list
    """
    clustering = nx.clustering(G, weight="trips")
    clustering = {k: v for k, v in sorted(clustering.items(), key=lambda item: item[1])}
    return list(reversed(clustering.keys()))


def get_communities(polygons, stations, trips):
    """
    Get communities of the demand in the city. The requests should
    consist the smallest possible administrative level for the city (e.g.
    districts or boroughs).
    :param polygons: Polygons of communities, keyed with name of community
    :type polygons: Dict of Shapely polygons
    :param stations: Stations of the demand
    :type stations: pandas dataframe
    :param trips:
    :type trips: pandas dataframe
    :return: Three dataframes, all, inner, inter
    :rtype: pandas dataframes
    """

    com_stat = {k: [] for k in polygons.keys()}
    stat_com = []
    for index, station in stations.iterrows():
        for com, poly in polygons.items():
            if poly.intersects(Point(station["longitude"], station["latitude"])):
                com_stat[com].append(station["station id"])
                stat_com.append(com)

    stations["community"] = stat_com

    com_trips = {k: 0 for k in product(polygons.keys(), polygons.keys())}
    for index, trip in trips.iterrows():
        s_com = stations.loc[stations["station id"] == trip["start station"]][
            "community"
        ].values[0]
        e_com = stations.loc[stations["station id"] == trip["end station"]][
            "community"
        ].values[0]
        com_trips[(s_com, e_com)] += trip["number of trips"]

    source = []
    source_inner = []
    source_inter = []
    target = []
    target_inner = []
    target_inter = []
    weight = []
    weight_inner = []
    weight_inter = []
    for k, v in com_trips.items():
        if v != 0:
            source.append(k[0])
            target.append(k[1])
            weight.append(v)
            if k[0] == k[1]:
                source_inner.append(k[0])
                target_inner.append(k[1])
                weight_inner.append(v)
            if k[0] != k[1]:
                source_inter.append(k[0])
                target_inter.append(k[1])
                weight_inter.append(v)

    trips = pd.DataFrame({"source": source, "target": target, "weight": weight})
    inner_trips = pd.DataFrame(
        {"source": source_inner, "target": target_inner, "weight": weight_inner}
    )
    inter_trips = pd.DataFrame(
        {"source": source_inter, "target": target_inter, "weight": weight_inter}
    )

    return trips, inner_trips, inter_trips


def stations_in_polygon(stations, polygon):
    """
    Checks which stations are inside the given polygon, only returns those.

    :param stations: Dict with stations {station_nbr: lat_long}
    :type stations: dict
    :param polygon: Shapely polygon of the area to check
    :type polygon: Shapely polygon
    :return: Dict with stations inside the polygon {station_nbr: lat_long}
    :rtype: dict
    """
    in_poly = {}
    for s_nbr, lat_long in stations.items():
        if polygon.intersects(Point(lat_long[1], lat_long[0])):
            in_poly[s_nbr] = lat_long
    return in_poly


def remove_stations(stations, g, nbr):
    """
    Removes the given number of stations, starting with the ones which are
    farthest away from their mapped nodes.

    :param stations: Dict with stations {station_nbr: lat_long}
    :type stations: dict
    :param g: Graph to map the stations to.
    :type g: networkx (Multi)(Di)graph
    :param nbr: Number of stations to delete.
    :type nbr: int
    :return: Dict with stations, without the removed stations
    :rtype: dict
    """
    s_lat = [p[0] for s, p in stations.items()]
    s_lon = [p[1] for s, p in stations.items()]
    s_nodes = list(ox.nearest_nodes(g, s_lon, s_lat))
    nodes_latlon = [(g.nodes[n]["y"], g.nodes[n]["x"]) for n in s_nodes]
    nodes_latlon = {s: nodes_latlon[idx] for idx, s in enumerate(stations.keys())}
    s_dist = {
        s: distance(
            stations[s][0],
            stations[s][1],
            nodes_latlon[s][0],
            nodes_latlon[s][1],
        )
        for s in stations.keys()
    }
    nbr = len(stations.keys()) - nbr
    rem = list(
        dict(sorted(s_dist.items(), key=itemgetter(1), reverse=True)[:nbr]).keys()
    )
    for k in rem:
        del stations[k]

    return stations


def average_hom_demand(
    save,
    comp_folder,
    bpp_base=None,
    rd_pattern="rd",
    nbr_of_rd_sets=10,
    mode="010",
):
    """
    Averages the bikeability for the randomised demand.
    :param save:
    :type save: str
    :param comp_folder: Storage folder of the result data for comparison.
    :type comp_folder: str
    :param bpp_base:
    :type bpp_base:
    :param rd_pattern: Naming scheme of the randomised demand.
    :type rd_pattern: str
    :param nbr_of_rd_sets: Number of data sets for the randomised demand.
    :type nbr_of_rd_sets: int
    :param mode:
    :type mode:
    :return: bpp and averaged ba
    """
    if bpp_base is None:
        bpp_min = []
        for rd_i in range(1, nbr_of_rd_sets + 1):
            data = h5py.File(
                f"{comp_folder}comp_{save}_{rd_pattern}_" f"{rd_i}.hdf5", "r"
            )["algorithm"]
            bpp_min.append(min(data[mode]["bpp complete"][()]))
        bpp_base = np.linspace(max(bpp_min), 1, num=10000)

    ba_c = []
    ends = []

    for rd_i in range(1, nbr_of_rd_sets + 1):
        data = h5py.File(f"{comp_folder}comp_{save}_{rd_pattern}_" f"{rd_i}.hdf5", "r")[
            "algorithm"
        ]
        f = interpolate.interp1d(data[mode]["bpp complete"][()], data[mode]["ba"][()])
        ba_c.append(f(bpp_base))
        ends.append(data[mode]["end"][()])
    ba = np.mean(np.array(ba_c), axis=0)
    return bpp_base, ba, max(ends)
