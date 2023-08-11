"""
This module includes all necessary functions for the data preparation and
handling.
"""
from ast import literal_eval
from os.path import join
from pathlib import Path
from .data_helper import *
from .plot import plot_used_nodes, plot_street_network


def prep_city(
    city_name,
    save_name,
    input_csv,
    nominatim_name=None,
    nominatim_result=1,
    trunk=False,
    consolidate=False,
    tol=35,
    by_bbox=True,
    by_city=True,
    by_polygon=True,
    cached_graph=False,
    cached_graph_file_path=None,
    paths=None,
    params=None,
):
    """
    Prepares the data of a city for the algorithm and saves it to the
    desired location.
    :param city_name: Name of the city
    :type city_name: str
    :param save_name: Savename of the city
    :type save_name: str
    :param input_csv: Path to the trip csv
    :type input_csv: str
    :param nominatim_name: Nominatim name of the city
    :type nominatim_name: str
    :param nominatim_result: results position of the city for the given name
    :type nominatim_result: int
    :param trunk: If trunks should be included or not
    :type trunk: bool
    :param consolidate: If intersections should be consolidated
    :type consolidate: bool
    :param tol: Tolerance of consolidation in meters
    :type tol: float
    :param by_bbox: If graph should be downloaded by the bbox surrounding the
    trips
    :type by_bbox: bool
    :param by_city: If graph should be downloaded by the nominatim name
    :type by_city: bool
    :param by_polygon: If graph should be downloaded by the given polygon
    :type  by_polygon: bool
    :param cached_graph: If a previously downloaded graph should be used.
    :type cached_graph: bool
    :param cached_graph_file_path: Path of the downloaded graph.
    :type cached_graph_file_path: str
    :param paths:
    :type paths:
    :param params: Dictionary with parameters for plotting etc
    :type params: dict or None
    :return: None
    """
    if paths is None:
        paths = create_default_paths()
    if params is None:
        params = create_default_params()

    output_folder = join(paths["input_folder"], save_name)
    plot_folder = join(paths["plot_folder"], "preparation", save_name)
    if paths["use_base_polygon"]:
        base_save = save_name.split(paths["save_devider"])[0]
        polygon_json = join(paths["polygon_folder"], f"{base_save}.json")
    else:
        polygon_json = join(paths["polygon_folder"], f"{save_name}.json")

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
                trunk=trunk,
                consolidate=consolidate,
                tol=tol,
                params=params,
            )
        else:
            print("Loading cached bbox map")
            G_b = ox.load_graphml(filepath=cached_graph_file_path)
            for u, v, data in G_b.edges(data=True):
                data["ex_inf"] = literal_eval(data["ex_inf"])
                data["turn_penalty"] = literal_eval(data["turn_penalty"])
            if consolidate:
                G_b = consolidate_nodes(G_b, tol=tol)

        # Loading trips inside bbox
        print("Mapping stations and calculation trips on map given by bbox")
        trips_b, stations_b = load_trips(G_b, input_csv)
        trips_b = trip_cyclist_type_split(
            trips_b, cyclist_split=params["cyclist_split"]
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

        save_graph(G_b, save_folder=output_folder, save_name=f"{save_name}_bbox")
        save_demand(trips_b, save_folder=output_folder, save_name=f"{save_name}_bbox")

    if by_city:
        if not cached_graph:
            # Download whole map of the city
            print("Downloading complete map of city")
            G_c = download_map_by_name(
                nominatim_name,
                nominatim_result,
                trunk=trunk,
                consolidate=consolidate,
                tol=tol,
                params=params,
            )
        else:
            print("Loading cached map of city")
            G_c = ox.load_graphml(filepath=cached_graph_file_path)
            for u, v, data in G_c.edges(data=True):
                data["ex_inf"] = literal_eval(data["ex_inf"])
                data["turn_penalty"] = literal_eval(data["turn_penalty"])
            if consolidate:
                G_c = consolidate_nodes(G_c, tol=tol)

        # Loading trips inside whole map
        print("Mapping stations and calculation trips on city map.")
        polygon_c = ox.geocode_to_gdf(nominatim_name, which_result=nominatim_result)
        trips_c, stations_c = load_trips(G_c, input_csv, polygon=polygon_c)
        trips_c = trip_cyclist_type_split(
            trips_c, cyclist_split=params["cyclist_split"]
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

        save_graph(G_c, save_folder=output_folder, save_name=f"{save_name}_city")
        save_demand(trips_c, save_folder=output_folder, save_name=f"{save_name}_city")

    if by_polygon:
        # Download cropped map (polygon)
        polygon = get_polygon_from_json(polygon_json)

        if not cached_graph:
            print("Downloading polygon.")
            G = download_map_by_polygon(
                polygon,
                trunk=trunk,
                consolidate=consolidate,
                tol=tol,
                params=params,
            )
        else:
            print("Loading cached map.")
            G = ox.load_graphml(filepath=cached_graph_file_path)
            for u, v, data in G.edges(data=True):
                data["ex_inf"] = literal_eval(data["ex_inf"])
                data["turn_penalty"] = literal_eval(data["turn_penalty"])
            if consolidate:
                G = consolidate_nodes(G, tol=tol)

        # Loading trips inside the polygon
        print("Mapping stations and calculation trips in polygon.")
        trips, stations = load_trips(G, input_csv, polygon=polygon)
        trips = trip_cyclist_type_split(trips, cyclist_split=params["cyclist_split"])

        # Colour all nodes used as origin or destination
        print("Plotting used nodes in polygon.")
        plot_used_nodes(
            city=city_name,
            save=save_name,
            G=G,
            trip_nbrs=trips,
            stations=stations,
            plot_folder=plot_folder,
        )
        # Plot used street network with all nodes
        plot_street_network(
            city=city_name,
            save=f"{save_name}",
            G=G,
            plot_folder=plot_folder,
            params=params,
        )
        save_graph(G, save_folder=output_folder, save_name=save_name)
        save_demand(trips, save_folder=output_folder, save_name=save_name)


def broaden_city_demand(
    city_name,
    save_name,
    graph_path,
    demand_path,
    radius,
    paths=None,
    params=None,
):
    if paths is None:
        paths = create_default_paths()
    if params is None:
        params = create_default_params()

    output_folder = join(paths["input_folder"], save_name)
    plot_folder = join(paths["plot_folder"], "preparation", save_name)

    G = ox.load_graphml(filepath=graph_path)
    demand = h5py.File(demand_path, "r")
    trips = {
        (int(k1), int(k2)): int(v[()])
        for k1 in list(demand.keys())
        for k2, v in demand[k1].items()
    }
    stations = [
        station for trip_id, nbr_of_trips in trips.items() for station in trip_id
    ]
    stations = list(set(stations))
    demand.close()

    save_graph(G, save_folder=output_folder, save_name=save_name)

    b_trips = broaden_demand(G, trips=trips, radius=radius)
    b_stations = [
        station for trip_id, nbr_of_trips in b_trips.items() for station in trip_id
    ]
    b_stations = list(set(b_stations))

    plot_used_nodes(
        city=city_name,
        save=save_name,
        G=G,
        trip_nbrs=b_trips,
        stations=b_stations,
        plot_folder=plot_folder,
    )

    b_demand = h5py.File(join(output_folder, f"{save_name}_demand.hdf5"), "w")
    b_demand.attrs["city"] = city_name
    b_demand.attrs["nbr of stations"] = len(b_stations)
    for k, v in b_trips.items():
        grp = b_demand.require_group(f"{k[0]}")
        grp[f"{k[1]}"] = v
    b_demand.close()
