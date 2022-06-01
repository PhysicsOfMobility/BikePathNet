"""
The core algorithm of the bikeability optimisation project.
"""
import h5py
import json
import time
import osmnx as ox
from pathlib import Path
from ..helper.algorithm_helper import *
from ..helper.logging_helper import log_to_file
from ..helper.setup_helper import create_default_params, create_default_paths


def core_algorithm(
    nkG,
    edge_dict,
    trips_dict,
    nk2nx_edges,
    street_cost,
    starttime,
    logpath,
    output_folder,
    save,
    dynamic,
    penalty,
):
    """

    :param nkG: Graph.
    :type nkG: networkit graph
    :param edge_dict: Dictionary of edges of G. {edge: edge_info}
    :type edge_dict: dict of dicts
    :param trips_dict: Dictionary with al information about the trips.
    :type trips_dict: dict of dicts
    :param nk2nx_edges: Dictionary that maps nk edges to nx edges.
    :type nk2nx_edges: dict
    :param street_cost: Dictionary with construction cost of street types
    :type street_cost: dict
    :param starttime: Time the script started. For logging only.
    :type starttime: timestamp
    :param logpath: Location of the log file
    :type logpath: str
    :param output_folder: Folder where the output should be stored.
    :type output_folder: str
    :param save: Save name of the network
    :type save: str
    :param dynamic: If penalties should be used for load weighting.
    :type dynamic: bool
    :param penalty: If trips should be recalculated each step.
    :type penalty: bool
    :return: None
    """
    # Initial calculation
    print("Initial calculation started.")
    calc_trips(nkG, edge_dict, trips_dict)
    print("Initial calculation ended.")

    # Initialise lists
    total_cost = [0]
    bike_path_perc = [bike_path_percentage(edge_dict)]
    total_real_distance_traveled = [total_len_on_types(trips_dict, "real")]
    total_felt_distance_traveled = [total_len_on_types(trips_dict, "felt")]
    nbr_on_street = [nbr_of_trips_on_street(trips_dict)]
    edited_edges = []
    edited_edges_nx = []

    log_at = [
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.025,
        0.01,
        0,
    ]
    log_idx = 0

    while True:
        # Calculate minimal loaded unedited edge:
        min_loaded_edge = get_minimal_loaded_edge(edge_dict, penalty=penalty)
        if min_loaded_edge == (-1, -1):
            print("Removed all bike paths.")
            break
        edited_edges.append(min_loaded_edge)
        edited_edges_nx.append(get_nx_edge(min_loaded_edge, nk2nx_edges))
        # Calculate cost of "adding" bike path
        total_cost.append(get_cost(min_loaded_edge, edge_dict, street_cost))
        # Edit minimal loaded edge and update edge_dict.
        edit_edge(nkG, edge_dict, min_loaded_edge)
        # Get all trips affected by editing the edge
        trips_recalc = {
            trip: trips_dict[trip] for trip in edge_dict[min_loaded_edge]["trips"]
        }

        # Recalculate all affected trips and update their information.
        calc_trips(nkG, edge_dict, trips_recalc, dynamic=dynamic)
        trips_dict.update(trips_recalc)
        # Store all important data
        bike_path_perc.append(bike_path_percentage(edge_dict))
        total_real_distance_traveled.append(total_len_on_types(trips_dict, "real"))
        total_felt_distance_traveled.append(total_len_on_types(trips_dict, "felt"))
        nbr_on_street.append(nbr_of_trips_on_street(trips_dict))

        # Logging
        next_log = log_at[log_idx]
        if bike_path_perc[-1] < next_log:
            log_to_file(
                file=logpath,
                txt=f"{save}: reached {next_log:3.2f} BPP",
                stamptime=time.localtime(),
                start=starttime,
                end=time.time(),
                stamp=True,
                difference=True,
            )
            loc = f"{output_folder}{save}_data.hdf5"
            hf = h5py.File(loc, "a")
            grp = hf.create_group(f"{log_idx+1:02d}")
            grp["ee_nk"] = edited_edges
            grp["ee_nx"] = edited_edges_nx
            grp["bpp"] = bike_path_perc
            grp["cost"] = total_cost
            grp["trdt"] = json.dumps(total_real_distance_traveled)
            grp["tfdt"] = json.dumps(total_felt_distance_traveled)
            grp["nos"] = nbr_on_street
            hf.close()
            log_idx += 1

    # Save data of this run to data array
    loc_hf = f"{output_folder}{save}_data.hdf5"
    hf = h5py.File(loc_hf, "a")
    grp = hf.create_group("all")
    grp["ee_nk"] = edited_edges
    grp["ee_nx"] = edited_edges_nx
    grp["bpp"] = bike_path_perc
    grp["cost"] = total_cost
    grp["trdt"] = json.dumps(total_real_distance_traveled)
    grp["tfdt"] = json.dumps(total_felt_distance_traveled)
    grp["nos"] = nbr_on_street
    hf.close()


def run_simulation(city, save, params=None, paths=None):
    """
    Prepares everything to run the core algorithm. All data will be saved to
    the given folders.
    :param city: Name of the city.
    :type city: str
    :param save: save name of everything associated with de place.
    :type save: str
    :param paths: Dictionary with folder paths (see example folder)
    :type paths: dict or None
    :param params: Dictionary with parameters (see example folder)
    :type params: dict or None
    :return: None
    """
    if paths is None:
        paths = create_default_paths()
    if params is None:
        params = create_default_params()

    input_folder = f'{paths["input_folder"]}{save}/'
    output_folder = f'{paths["output_folder"]}{save}/'
    log_folder = f'{paths["log_folder"]}{save}/'
    # Check if necessary folders exists, otherwise create.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    loc_hf = f"{output_folder}{save}_data.hdf5"
    hf = h5py.File(loc_hf, "w")
    hf.attrs["city"] = city

    # Start date and time for logging.
    sd = time.localtime()
    starttime = time.time()

    logpath = f"{log_folder}{save}.txt"
    # Initial Log
    log_to_file(logpath, f"Started {save}", start=sd, stamp=False, difference=False)

    nxG = ox.load_graphml(
        filepath=f"{input_folder}{save}.graphml", node_dtypes={"street_count": float}
    )
    nxG = nx.Graph(nxG.to_undirected())
    print(f"Starting with {save}")
    hf.attrs["nodes"] = len(nxG.nodes)
    hf.attrs["edges"] = len(nxG.edges)

    demand = h5py.File(f"{input_folder}{save}_demand.hdf5", "r")
    trip_nbrs_nx = {
        (int(k1), int(k2)): int(v[()])
        for k1 in list(demand.keys())
        for k2, v in demand[k1].items()
    }
    demand.close()

    stations = [
        station for trip_id, nbr_of_trips in trip_nbrs_nx.items() for station in trip_id
    ]
    stations = list(set(stations))
    hf.attrs["nbr of stations"] = len(stations)
    hf["stations"] = stations

    hf.attrs["total trips (incl round trips)"] = sum(trip_nbrs_nx.values())

    # Exclude round trips
    trip_nbrs_nx = {
        trip_id: nbr_of_trips
        for trip_id, nbr_of_trips in trip_nbrs_nx.items()
        if not trip_id[0] == trip_id[1]
    }
    hf.attrs["total trips"] = sum(trip_nbrs_nx.values())

    # Convert networkx graph into network kit graph
    nkG = nk.nxadapter.nx2nk(nxG, weightAttr="length")
    nkG.removeSelfLoops()

    # Setup mapping dictionaries between nx and nk
    nx2nk_nodes = {list(nxG.nodes)[n]: n for n in range(len(list(nxG.nodes)))}
    nk2nx_nodes = {v: k for k, v in nx2nk_nodes.items()}
    nx2nk_edges = {
        (e[0], e[1]): (nx2nk_nodes[e[0]], nx2nk_nodes[e[1]]) for e in list(nxG.edges)
    }
    nk2nx_edges = {v: k for k, v in nx2nk_edges.items()}

    # Trips dict for the nk graph
    trip_nbrs_nk = {
        (nx2nk_nodes[k[0]], nx2nk_nodes[k[1]]): v for k, v in trip_nbrs_nx.items()
    }

    # All street types in network
    street_types = ["primary", "secondary", "tertiary", "residential"]

    # Setup length on street type dict
    len_on_type = {t: 0 for t in street_types}
    len_on_type["bike path"] = 0

    # Set penalties for different street types
    penalties = params["penalties"]

    # Set cost for different street types
    street_cost = params["street_cost"]

    # Setup trips and edge dict
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
        for t_id, nbr_of_trips in trip_nbrs_nk.items()
    }
    edge_dict = {
        edge: {
            "felt length": get_street_length(nxG, edge, nk2nx_edges),
            "real length": get_street_length(nxG, edge, nk2nx_edges),
            "street type": get_street_type_cleaned(nxG, edge, nk2nx_edges),
            "penalty": penalties[get_street_type_cleaned(nxG, edge, nk2nx_edges)],
            "bike path": True,
            "load": 0,
            "trips": [],
            "original edge": nk2nx_edges[edge],
        }
        for edge in nkG.iterEdges()
    }

    hf.close()
    # Calculate data
    core_algorithm(
        nkG=nkG,
        edge_dict=edge_dict,
        trips_dict=trips_dict,
        nk2nx_edges=nk2nx_edges,
        street_cost=street_cost,
        starttime=starttime,
        logpath=logpath,
        output_folder=output_folder,
        save=save,
        penalty=params["penalty weighting"],
        dynamic=params["dynamic routes"],
    )

    # Print computation time to console and write it to the log.
    log_to_file(
        logpath,
        f"Finished {save}.",
        stamptime=time.localtime(),
        start=starttime,
        end=time.time(),
        stamp=True,
        difference=True,
    )
    return 0
