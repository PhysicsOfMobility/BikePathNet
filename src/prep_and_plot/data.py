"""
This module includes all necessary functions for the data preparation and handling.
"""
from os.path import join
from pathlib import Path
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
    trunk: bool = False,
    consolidate: bool = False,
    tol=35,
    by_bbox: bool = True,
    by_city: bool = True,
    by_polygon: bool = True,
    cached_graph: bool = False,
    cached_graph_file_path: str | None = None,
    params: dict | None = None,
    paths: dict | None = None,
):
    """Prepares the data of a city for the algorithm and saves it to the desired location.

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
    trunk : bool
        If trunks should be included or not (Default value = False)
    consolidate : bool
        If intersections should be consolidated (Default value = False)
    tol : float
        Tolerance of consolidation in meters (Default value = 35)
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
        Dict with the params for plots etc., check 'setup_params.py' in the 'scripts' folder. (Default value = None)
    paths : dict | None
        Dict with the paths for plots etc., check 'setup_paths.py' in the 'scripts' folder. (Default value = None)

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
                data["turn_penalty"] = {int(key): value for key, value in data["turn_penalty"].items()}
                data["cost"] = literal_eval(data["cost"])
            if consolidate:
                G_b = consolidate_nodes(G_b, tol=tol)

        # Loading trips inside bbox
        print("Mapping stations and calculation trips on map given by bbox")
        trips_b, stations_b = load_trips(G_b, input_csv)
        trips_b = trip_cyclist_type_split(trips_b, cyclist_split=params["cyclist_split"])
        print(
            f"Number of Stations: {len(stations_b)}, "
            f"Number of trips: {sum([sum(t_b.values()) for t_b in trips_b.values()])} "
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
                data["turn_penalty"] = {int(key): value for key, value in data["turn_penalty"].items()}
                data["cost"] = literal_eval(data["cost"])
            if consolidate:
                G_c = consolidate_nodes(G_c, tol=tol)

        # Loading trips inside whole map
        print("Mapping stations and calculation trips on city map.")
        polygon_c = ox.geocode_to_gdf(
            nominatim_name, which_result=nominatim_result
        )
        trips_c, stations_c = load_trips(G_c, input_csv, polygon=polygon_c)
        trips_c = trip_cyclist_type_split(trips_c, cyclist_split=params["cyclist_split"])
        print(
            f"Number of Stations: {len(stations_c)}, "
            f"Number of trips: {sum([sum(t_c.values()) for t_c in trips_c.values()])} "
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
                data["turn_penalty"] = {int(key): value for key, value in data["turn_penalty"].items()}
                data["cost"] = literal_eval(data["cost"])
            if consolidate:
                G = consolidate_nodes(G, tol=tol)

        # Loading trips inside the polygon
        print("Mapping stations and calculation trips in polygon.")
        trips, stations = load_trips(G, input_csv, polygon=polygon)
        trips = trip_cyclist_type_split(trips, cyclist_split=params["cyclist_split"])
        print(
            f"Number of Stations: {len(stations)}, "
            f"Number of trips: {sum([sum(t.values()) for t in trips.values()])}, "
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
