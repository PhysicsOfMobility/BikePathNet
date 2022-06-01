"""
This module includes all necessary helper functions for the data preparation
and handling.
"""
import h5py
import json
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
from scipy import interpolate
from operator import itemgetter
from math import cos, asin, sqrt, pi
from shapely.geometry import Point, Polygon
from .algorithm_helper import get_street_type, calc_single_state
from .setup_helper import create_default_params, create_default_paths


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
            lambda row: row["start in polygon"] and row["end in polygon"], axis=1
        )
        start_lat = list(trips.loc[trips["in polygon"]]["start latitude"])
        start_long = list(trips.loc[trips["in polygon"]]["start longitude"])
        end_lat = list(trips.loc[trips["in polygon"]]["end latitude"])
        end_long = list(trips.loc[trips["in polygon"]]["end longitude"])
        nbr_of_trips = list(trips.loc[trips["in polygon"]]["number of trips"])
        return nbr_of_trips, start_lat, start_long, end_lat, end_long


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

    nbr_of_trips, start_lat, start_long, end_lat, end_long = get_lat_long_trips(
        path_to_trips, polygon, delim=delim
    )

    start_nodes = list(ox.nearest_nodes(G, start_long, start_lat))
    end_nodes = list(ox.nearest_nodes(G, end_long, end_lat))

    trip_nbrs = {}
    for trip in range(len(nbr_of_trips)):
        trip_nbrs[(int(start_nodes[trip]), int(end_nodes[trip]))] = int(
            nbr_of_trips[trip]
        )

    stations = set()
    for k, v in trip_nbrs.items():
        stations.add(k[0])
        stations.add(k[1])

    trip_nbrs = {k: v for k, v in trip_nbrs.items() if not k[0] == k[1]}
    return trip_nbrs, stations


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


def consolidate_nodes(G, tol):
    """
    Consolidates intersections of graph g with given tolerance in meters.
    :param G: Graph to consolidate intersections in
    :type G: networkx (Multi)(Di)Graph
    :param tol: Tolerance for consolidation in meters
    :type tol: float or int
    :return: Graph with consolidated intersections
    :rtype same as param G
    """
    H = ox.project_graph(G, to_crs="epsg:2955")
    H = ox.consolidate_intersections(
        H, tolerance=tol, rebuild_graph=True, dead_ends=True, reconnect_edges=True
    )
    print(
        "Consolidating intersections. Nodes before: {}. Nodes after: {}".format(
            len(G.nodes), len(H.nodes)
        )
    )
    H = nx.convert_node_labels_to_integers(H)
    nx.set_node_attributes(H, {n: n for n in H.nodes}, "osmid")
    G = ox.project_graph(H, to_crs="epsg:4326")
    # Bugfix for node street_count is set as gloat instead of int
    sc = {
        n: int(G.nodes[n]["street_count"])
        for n in G.nodes
        if "street_count" in G.nodes[n].keys()
    }
    nx.set_node_attributes(G, sc, "street_count")
    return G


def prepare_downloaded_map(G, consolidate=False, tol=35):
    """
    Prepares the downloaded map. Removes all motorway edges and trunk edges.
    Turns it to undirected, removes all isolated nodes using networkxs
    isolates() function and reduces the graph to the greatest connected
    component using osmnxs get_largest_component().
    :param G: Graph to clean.
    :type G: networkx graph
    :param consolidate: Set true if intersections should bes consolidated.
    :type consolidate: bool
    :param tol: Tolerance of intersection consolidation in meters
    :type tol: float
    :return: Cleaned graph
    :rtype: networkx graph.
    """
    # Remove self loops
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print("Removed {} self loops.".format(len(self_loops)))

    # Remove motorways and trunks
    s_t = ["motorway", "motorway_link", "trunk", "trunk_link"]
    edges_to_remove = [e for e in G.edges() if get_street_type(G, e, multi=True) in s_t]
    G.remove_edges_from(edges_to_remove)
    print("Removed {} car only edges.".format(len(edges_to_remove)))

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print("Removed {} isolated nodes.".format(len(isolated_nodes)))
    G = ox.utils_graph.get_largest_component(G)
    print("Reduce to largest connected component")

    if consolidate:
        G = consolidate_nodes(G, tol)

    # Bike graph assumed undirected.
    G = G.to_undirected()
    print("Turned graph to undirected.")

    return G


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
            stations[s][0], stations[s][1], nodes_latlon[s][0], nodes_latlon[s][1]
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
    bpp_base=np.linspace(0, 1, num=10000),
    hom_pattern="hom",
    nbr_of_hom_sets=10,
):
    """
    Averages the bikeability for the randomised demand.
    :param save:
    :type save: str
    :param comp_folder: Storage folder of the result data for comparison.
    :type comp_folder: str
    :param bpp_base:
    :type bpp_base:
    :param hom_pattern: Naming scheme of the randomised demand.
    :type hom_pattern: str
    :param nbr_of_hom_sets: Number of data sets for the randomised demand.
    :type nbr_of_hom_sets: int
    :return: bpp and averaged ba
    """
    ba_c = []

    for hom_i in range(nbr_of_hom_sets):
        data = h5py.File(
            f"{comp_folder}comp_{save}_{hom_pattern}_" f"{hom_i+1}.hdf5", "r"
        )["algorithm"]
        f = interpolate.interp1d(data["bpp"][()], data["ba"][()])
        ba_c.append(f(bpp_base))
    ba = np.mean(np.array(ba_c), axis=0)
    return bpp_base, ba


def calc_average_trip_len(nxG, trip_nbrs, penalties=True, params=None):
    """
    Calculate the average trip length of given trips in given graph.
    :param nxG: Graph to calculate trips in
    :type nxG: networkx graph
    :param trip_nbrs: Trips
    :type trip_nbrs: dict
    :param penalties: If penalties should be applied or not
    :type penalties: bool
    :param params:
    :type params: None or dict
    :return: Average trip length in meters
    :rtype: float
    """
    if params is None:
        params = create_default_params()

    if penalties:
        bike_paths = []
    else:
        bike_paths = list(nxG.edges())

    nxG = nx.Graph(nxG.to_undirected())
    trips_dict = calc_single_state(
        nxG, trip_nbrs, bike_paths=bike_paths, params=params
    )[7]

    length = []
    for trip, trip_info in trips_dict.items():
        length += [trip_info["length felt"]] * trip_info["nbr of trips"]
    return np.average(length)
