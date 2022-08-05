"""
This module includes all necessary helper functions for the main algorithm.
"""
import networkit as nk
import networkx as nx
import numpy as np
from .setup_helper import create_default_params, create_default_paths


def get_street_type(G, edge, nk2nx=None, multi=False):
    """
    Returns the street type of the edge in G. If 'highway' in G is al list,
    return first entry.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Street type.
    :rtype: str
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    if multi:
        street_type = G[edge[0]][edge[1]][0]["highway"]
    else:
        street_type = G[edge[0]][edge[1]]["highway"]
    if isinstance(street_type, str):
        return street_type
    else:
        return street_type[0]


def get_street_type_cleaned(G, edge, nk2nx=None, multi=False):
    """
    Returns the street type of the edge. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Street type·
    :rtype: str
    """
    st = get_street_type(G, edge, nk2nx, multi=multi)
    if st in ["primary", "primary_link", "trunk", "trunk_link"]:
        return "primary"
    elif st in ["secondary", "secondary_link"]:
        return "secondary"
    elif st in ["tertiary", "tertiary_link"]:
        return "tertiary"
    else:
        return "residential"


def get_all_street_types(G, multi=False):
    """
    Returns all street types appearing in G.
    :param G: Graph.
    :type G: networkx graph.
    :param multi: Set True if G is a MultiGraph
    :type multi: bool
    :return: List of all street types.
    :rtype: list of str
    """
    street_types = set()
    for edge in G.edges():
        street_types.add(get_street_type(G, edge, multi=multi))
    return list(street_types)


def get_all_street_types_cleaned(G, multi=False):
    """
    Returns all street types appearing in G. Street types are reduced to
    primary, secondary, tertiary and residential.
    :param G: Graph.
    :type G: networkx graph.
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: List of all street types.
    :rtype: list of str
    """
    street_types_cleaned = set()
    for edge in G.edges():
        street_types_cleaned.add(get_street_type(G, edge, multi=multi))
    return list(street_types_cleaned)


def get_street_length(G, edge, nk2nx=None, multi=False):
    """
    Returns the length of the edge in G.
    :param G: Graph.
    :type G: networkx graph.
    :param edge: Edge in the networkx or networkit graph. If edge is from the
    networkit graph, nk2nx dict has to be given, if its from the networkx
    graph nk2nx has to be set False.
    :type edge: tuple of integers
    :param nk2nx: edge map form networkit graph to networkx graph.
    :type nk2nx: dict
    :param multi: Set true if G is a MultiGraph
    :type multi: bool
    :return: Length of edge.
    :rtype: float
    """
    if isinstance(nk2nx, dict):
        if edge in nk2nx:
            edge = nk2nx[edge]
        else:
            edge = nk2nx[(edge[1], edge[0])]
    if multi:
        length = G[edge[0]][edge[1]][0]["length"]
    else:
        length = G[edge[0]][edge[1]]["length"]
    return length


def get_cost(edge, edge_dict, cost_dict):
    """
    Returns the cost of an edge depending on its street type.
    :param edge: Edge.
    :type edge: tuple of integers
    :param edge_dict: Dictionary with all edge information.
    :type edge_dict: dict of dicts
    :param cost_dict: Dictionary with cost of edge depending on street type.
    :type cost_dict: dict
    :return: Cost of the edge
    :rtype: float
    """
    street_type = edge_dict[edge]["street type"]
    street_length = edge_dict[edge]["real length"]
    return street_length * cost_dict[street_type]


def get_total_cost(bikepaths, edge_dict, cost_dict):
    """
    Returns the cost of building bike paths.
    :param bikepaths: Edges with bike paths.
    :type bikepaths: list of tuple of int
    :param edge_dict: Dictionary with all edge information.
    :type edge_dict: dict of dicts
    :param cost_dict: Dictionary with cost of edge depending on street type.
    :type cost_dict: dict
    :return: Cost of the edge
    :rtype: float
    """
    total_cost = 0
    for edge in bikepaths:
        total_cost += get_cost(edge, edge_dict, cost_dict)
    return total_cost


def get_trip_edges(edges_dict, trip_nodes):
    """
    Returns the edge sequence of a trip given by its node sequence.
    :param edges_dict: Dictionary with all information about the edges.
    :type edges_dict: Dict of dicts.
    :param trip_nodes: Node sequence of a trip.
    :type trip_nodes: list of integers
    :return: Edge sequence.
    :rtype: list of tuples of integers
    """
    edge_sequence = []
    for i in range(len(trip_nodes) - 1):
        f_n = trip_nodes[i]  # First node
        s_n = trip_nodes[i + 1]  # Second node of the edge
        # Dict doesn't accept (2, 1) for undirected edge (1, 2):
        if (f_n, s_n) in edges_dict:
            edge_sequence.append((f_n, s_n))
        else:
            edge_sequence.append((s_n, f_n))
    return edge_sequence


def get_minimal_loaded_edge(edge_dict, penalty=True, forward=False):
    """
    Returns the minimal loaded edge in edge_list.
    If unedited=True it returns the minimal loaded unedited edge.
    If there are multiple edges with the same minimal load, one is randomly
    drawn.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts.
    :param penalty: If penalties should be used for load weighting.
    :type penalty: bool
    :param forward: Starting from scratch and adding bike paths
    :type forward: bool
    :return: minimal loaded edge
    :rtype: Tuple of integers
    """
    if forward:
        edges = {
            edge: edge_info
            for edge, edge_info in edge_dict.items()
            if not edge_info["bike path"]
        }
    else:
        edges = {
            edge: edge_info
            for edge, edge_info in edge_dict.items()
            if edge_info["bike path"]
        }

    if penalty:
        if forward:
            edges_load = {
                edge: edge_info["load"] * (1 / edge_info["penalty"])
                for edge, edge_info in edges.items()
            }
        else:
            edges_load = {
                edge: edge_info["load"] * edge_info["penalty"]
                for edge, edge_info in edges.items()
            }
    else:
        edges_load = {edge: edge_info["load"] for edge, edge_info in edges.items()}

    if edges_load == {}:
        return (-1, -1)
    else:
        if forward:
            max_load = max(edges_load.values())
            max_edges = [e for e, load in edges_load.items() if load == max_load]
            max_edge = max_edges[np.random.choice(len(max_edges))]
            return max_edge
        else:
            min_load = min(edges_load.values())
            min_edges = [e for e, load in edges_load.items() if load == min_load]
            min_edge = min_edges[np.random.choice(len(min_edges))]
            return min_edge


def bike_path_percentage(edge_dict):
    """
    Returns the bike path percentage by length.
    :param edge_dict: Dictionary with all information about the edges
    :type edge_dict: dict of dicts
    :return: percentage of bike paths by length.
    :rtype float
    """
    bike_length = 0
    total_length = 0
    for edge, edge_info in edge_dict.items():
        total_length += edge_info["real length"]
        if edge_info["bike path"]:
            bike_length += edge_info["real length"]
    return bike_length / total_length


def check_if_trip_on_street(trip_info, edge_dict):
    """
    Checks if given trip is somewhere on a street.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: True if on street false if not.
    :rtype: bool
    """
    for edge in trip_info["edges"]:
        if not edge_dict[edge]["bike path"]:
            return True
    return False


def nbr_of_trips_on_street(trips_dict):
    """
    Returns the number of trips that are somewhere on a street.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: Number of trips at least once on a street.
    :rtype: integer
    """
    nbr_on_street = 0
    for trip, trip_info in trips_dict.items():
        if trip_info["on street"]:
            nbr_on_street += trip_info["nbr of trips"]
    return nbr_on_street


def set_trips_on_street(trips_dict, edge_dict):
    """
    Sets "on street" value in trips_dict to the right value.
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips_dict.
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info["on street"] = False
        for edge in trip_info["edges"]:
            if not edge_dict[edge]["bike path"]:
                trip_info["on street"] = True
    return trips_dict


def get_len_of_trips_over_edge(edge, edge_list, trips_dict):
    """
    Returns the total traveled distance over the given edge.
    ttd = edge length * nbr of trips over edge
    :param edge: Edge.
    :type edge: tuple of integers
    :param edge_list: Dictionary with all information about the edges.
    :type edge_list: dict of dicts
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :return: Total traveled distance.
    :rtype float
    """
    length = 0
    for trip in edge_list[edge]["trips"]:
        length += trips_dict[trip]["nbr of trips"] * trips_dict[trip]["length real"]
    return length


def real_trip_length(trip_info, edge_dict):
    """
    Returns the real length og a trip.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict of dicts.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Real length of the trip.
    :rtype: float
    """
    length = sum([edge_dict[edge]["real length"] for edge in trip_info["edges"]])
    return length


def felt_trip_length(trip_info, edge_dict):
    """
    Returns the felt length og a trip.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict of dicts.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Felt length of the trip.
    :rtype: float
    """
    length = sum([edge_dict[edge]["felt length"] for edge in trip_info["edges"]])
    return length


def len_on_types(trip_info, edge_dict, len_type="real"):
    """
    Returns a dict with the length of the trip on the different street types.
    len_type defines if felt or real length.
    :param trip_info: Dictionary with all information about the trip.
    :type trip_info: dict
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param len_type: 'real' or 'felt' length is used.
    :type len_type: str
    :return: Dictionary with length on different street types.
    :rtype: dict
    """
    len_on_type = {t: 0 for t, l in trip_info[len_type + " length on types"].items()}
    for edge in trip_info["edges"]:
        street_type = edge_dict[edge]["street type"]
        street_length = edge_dict[edge][len_type + " length"]
        if edge_dict[edge]["bike path"]:
            len_on_type["bike path"] += street_length
        else:
            len_on_type[street_type] += street_length
    return len_on_type


def total_len_on_types(trips_dict, len_type):
    """
    Returns the total distance driven sorted by street type.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param len_type: 'real' or 'felt' length is used.
    :type len_type: str
    :return: Dictionary with total length on different street types.
    :rtype: dict
    """
    lop = 0
    los = 0
    lot = 0
    lor = 0
    lob = 0
    for trip, trip_info in trips_dict.items():
        nbr_of_trips = trip_info["nbr of trips"]
        lop += nbr_of_trips * trip_info[len_type + " length on types"]["primary"]
        los += nbr_of_trips * trip_info[len_type + " length on types"]["secondary"]
        lot += nbr_of_trips * trip_info[len_type + " length on types"]["tertiary"]
        lor += nbr_of_trips * trip_info[len_type + " length on types"]["residential"]
        lob += nbr_of_trips * trip_info[len_type + " length on types"]["bike path"]
    tlos = lop + los + lot + lor
    tloa = tlos + lob
    return {
        "total length on all": tloa,
        "total length on street": tlos,
        "total length on primary": lop,
        "total length on secondary": los,
        "total length on tertiary": lot,
        "total length on residential": lor,
        "total length on bike paths": lob,
    }


def set_len(trips_dict, edge_dict):
    """
    Sets the length of a trip to the correct value in the trips dictionary.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param edge_dict:  Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips dictionary
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info["length real"] = real_trip_length(trip_info, edge_dict)
        trip_info["length felt"] = felt_trip_length(trip_info, edge_dict)
    return trips_dict


def set_len_on_types(trips_dict, edge_dict):
    """
    Sets the length by type of a trip to the correct value in the trips
    dictionary.
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param edge_dict:  Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :return: Updated trips dictionary
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        trip_info["real length on types"] = len_on_types(trip_info, edge_dict, "real")
        trip_info["felt length on types"] = len_on_types(trip_info, edge_dict, "felt")
    return trips_dict


def add_load(edge_dict, trips_dict):
    """
    Adds load and trip_id of the given trips to the edges.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :return: edge_dict with updated information
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        for e in trip_info["edges"]:
            edge_dict[e]["trips"] += [trip]
            edge_dict[e]["load"] += trip_info["nbr of trips"]
    return edge_dict


def delete_load(edge_dict, trips_dict, dynamic=True):
    """
    Deletes load and trip_id of the given trips from the edges.
    :param edge_dict: Dictionary with all information about the edges.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trip.
    :type trips_dict: dict of dicts
    :param dynamic: If penalties should be used for load weighting.
    :type dynamic: bool
    :return: edge_dict with updated information
    :rtype: dict of dicts
    """
    for trip, trip_info in trips_dict.items():
        for e in trip_info["edges"]:
            edge_dict[e]["trips"].remove(trip)
            if dynamic:
                edge_dict[e]["load"] -= trip_info["nbr of trips"]
    return edge_dict


def get_all_shortest_paths(G, source):
    """
    Returns all shortest paths with sources definde in sources.
    :param G: Graph.
    :type G: networkit graph
    :param source: Source node to calculate the shortest paths.
    :type source: int
    :return: Dict with all shortest paths, keyed by target.
    :rtype: dict
    """
    d = nk.distance.Dijkstra(G, source, storePaths=True)
    d.run()
    shortest_paths = {tgt: d.getPath(tgt) for tgt in list(G.iterNodes())}
    return shortest_paths


def get_nx_edge(nk_edge, nk2nx_edges):
    """
    Returns the networkx edge for the given networkit edge.
    :param nk_edge: Edge in the nk graph.
    :type nk_edge: tuple
    :param nk2nx_edges: Dict mapping nk edges to nx edges.
    :type nk2nx_edges: dict
    :return: Edge in the nx graph.
    :rtype: tuple
    """
    if nk_edge in nk2nx_edges:
        return nk2nx_edges[nk_edge]
    else:
        return nk2nx_edges[(nk_edge[1], nk_edge[0])]


def convert_edge_list(edge_list, edge_dict):
    new_edge_list = [edge_dict[e]["original edge"] for e in edge_list]
    return new_edge_list


def set_sp_info(source, shortest_paths, edge_dict, trips_dict, dynamic=True):
    """
    Set info of the shortest paths to  trip and edge dicts.
    :param source: Source node
    :type source: int
    :param shortest_paths: Dict of shortest paths.
    :type shortest_paths: dict
    :param edge_dict: Edge dict
    :type edge_dict: dict
    :param trips_dict: Trips dict
    :type trips_dict: dict
    :param dynamic: If penalties should be used for load weighting.
    :type dynamic: bool
    :return: None
    """
    for trip, trip_info in trips_dict.items():
        if trip[0] == source:
            if not trip_info["nodes"] == shortest_paths[trip[1]]:
                delete_load(edge_dict, {trip: trip_info}, dynamic=dynamic)
                trip_info["nodes"] = shortest_paths[trip[1]]
                trip_info["edges"] = get_trip_edges(edge_dict, trip_info["nodes"])
                for e in trip_info["edges"]:
                    edge_dict[e]["trips"] += [trip]
                    if dynamic:
                        edge_dict[e]["load"] += trip_info["nbr of trips"]
            trip_info["length felt"] = felt_trip_length(trip_info, edge_dict)
            trip_info["length real"] = real_trip_length(trip_info, edge_dict)
            trip_info["real length on types"] = len_on_types(
                trip_info, edge_dict, "real"
            )
            trip_info["felt length on types"] = len_on_types(
                trip_info, edge_dict, "felt"
            )
            trip_info["on street"] = check_if_trip_on_street(trip_info, edge_dict)


def calc_trips(G, edge_dict, trips_dict, netwx=False, dynamic=True):
    """
    Calculates the shortest paths for all trips and sets all corresponding
    information in the trip_dict
    :param G: graph to calculate the s.p. in.
    :type G: networkit or networkx graph
    :param edge_dict: Dictionary with all information about the edges of G.
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with all information about the trips.
    :type trips_dict: dict of dicts
    :param netwx: If G is a networkx graph set netwx to True.
    :type netwx: bool
    :param dynamic: If penalties should be used for load weighting.
    :type dynamic: bool
    :return: Updated trips_dict and edge_dict.
    :rtype: dict of dicts
    """
    # Calculate single source paths for all origin nodes
    origin_nodes = list({k[0] for k, v in trips_dict.items()})
    if not netwx:
        for source in origin_nodes:
            shortest_paths = get_all_shortest_paths(G, source)
            # Set all information to trip_info and edge_info
            set_sp_info(source, shortest_paths, edge_dict, trips_dict, dynamic=dynamic)
    else:
        for source in origin_nodes:
            shortest_paths = nx.single_source_dijkstra_path(G, source, weight="length")
            # Set all information to trip_info and edge_info
            set_sp_info(source, shortest_paths, edge_dict, trips_dict, dynamic=dynamic)
    return trips_dict, edge_dict


def edit_edge(nkG, edge_dict, edge):
    """
    Edits "felt length" of given edge  in the edge_dict and "length" in G.
    Length change is done corresponding to the street type of the edge.
    :param nkG: Graph.
    :type nkG: networkit graph
    :param edge_dict: Dictionary with all information about the edges of G.
    :type edge_dict: dict of dicts
    :param edge: Edge to edit.
    :type edge: tuple of integers
    :return: Updated G and edge_dict.
    :rtype: networkx graph and  dict of dicts
    """
    edge_dict[edge]["bike path"] = not edge_dict[edge]["bike path"]
    edge_dict[edge]["felt length"] *= edge_dict[edge]["penalty"]
    nkG.setWeight(edge[0], edge[1], edge_dict[edge]["felt length"])
    return nkG, edge_dict


def calc_single_state(nxG, trip_nbrs, bike_paths, params=None):
    """
    Calculates the data for the current bike path situation. If no bike
    paths are provided all primary and secondary roads will be assigned with a
    bike path.
    :param nxG: Street graph to calculate in.
    :param trip_nbrs: Number of trips as used for the main algorithm.
    :param bike_paths:
    :param params:
    :return: Data structured as from the main algorithm.
    :rtype: np.array
    """
    if params is None:
        params = create_default_params()

    # All street types in network
    len_on_type = {
        t: 0 for t in ["primary", "secondary", "tertiary", "residential", "bike path"]
    }

    # Set penalties for different street types
    penalties = params["penalties"]

    # Set cost for different street types
    street_cost = params["street_cost"]

    trips_dict = {
        t_id: {
            "nbr of trips": nbr_of_trips,
            "nodes": [],
            "edges": [],
            "length real": 0,
            "length felt": 0,
            "real length on types": len_on_type,
            "felt length on types": len_on_type,
            "on street": False,
        }
        for t_id, nbr_of_trips in trip_nbrs.items()
    }
    edge_dict = {
        edge: {
            "felt length": get_street_length(nxG, edge),
            "real length": get_street_length(nxG, edge),
            "street type": get_street_type_cleaned(nxG, edge),
            "penalty": penalties[get_street_type_cleaned(nxG, edge)],
            "bike path": True,
            "load": 0,
            "trips": [],
        }
        for edge in nxG.edges()
    }

    for edge, edge_info in edge_dict.items():
        if edge not in bike_paths:
            edge_info["bike path"] = False
            edge_info["felt length"] *= edge_info["penalty"]
            nxG[edge[0]][edge[1]]["length"] *= edge_info["penalty"]

    calc_trips(nxG, edge_dict, trips_dict, netwx=True)

    # Initialise lists
    total_cost = get_total_cost(bike_paths, edge_dict, street_cost)
    bike_path_perc = bike_path_percentage(edge_dict)
    total_real_distance_traveled = total_len_on_types(trips_dict, "real")
    total_felt_distance_traveled = total_len_on_types(trips_dict, "felt")
    nbr_on_street = nbr_of_trips_on_street(trips_dict)

    # Save data of this run to data array
    data = np.array(
        [
            bike_paths,
            total_cost,
            bike_path_perc,
            total_real_distance_traveled,
            total_felt_distance_traveled,
            nbr_on_street,
            edge_dict,
            trips_dict,
        ],
        dtype=object,
    )
    return data
