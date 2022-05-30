"""
This module includes all necessary functions for the data preparation and
handling.
"""
import itertools as it
from pathlib import Path
from rpy2.robjects import r, pandas2ri
from ..helper.data_helper import *
from ..helper.setup_helper import create_default_params, create_default_paths


def prep_city(city_name, save_name,  input_csv, consolidate=False, tol=35,
              cached_graph=False, cached_graph_folder=None,
              cached_graph_name=None, paths=None, params=None):
    """
    Prepares the data of a city for the algorithm and saves it to the
    desired location.
    :param city_name: Name of the city
    :type city_name: str
    :param save_name: Save name of the city
    :type save_name: str
    :param input_csv: Path to the trip csv
    :type input_csv: str
    :param consolidate: If intersections should be consolidated
    :type consolidate: bool
    :param tol: Tolerance of consolidation in meters
    :type tol: float
    :param cached_graph: If a previously downloaded graph should be used.
    :type cached_graph: bool
    :param cached_graph_folder: Folder of the downloaded graph.
    :type cached_graph_folder: str
    :param cached_graph_name: Name of the downloaded graph.
    :type cached_graph_name: str
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

    input_csv = f'{paths["csv_folder"]}{input_csv}'
    output_folder = f'{paths["input_folder"]}{save_name}/'
    plot_folder = f'{paths["plot_folder"]}preparation/'
    polygon_json = f'{paths["polygon_folder"]}{save_name}.json'

    # Check if necessary folders exists, otherwise create.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

    if cached_graph_folder is None:
        cached_graph_folder = output_folder
    if cached_graph is None:
        cached_graph_name = save_name

    # Download cropped map (polygon)
    polygon = get_polygon_from_json(polygon_json)

    if not cached_graph:
        print(f'Downloading graph for {city_name}.')
        G = ox.graph_from_polygon(polygon, network_type='drive')
        G = prepare_downloaded_map(G, consolidate=consolidate, tol=tol)
    else:
        print(f'Loading cached map for {city_name}.')
        G = ox.load_graphml(filepath=f'{cached_graph_folder}'
                                     f'{cached_graph_name}.graphml')
        if consolidate:
            G = consolidate_nodes(G, tol=tol)

    # Loading trips inside the polygon
    print('Mapping stations and calculation trips in polygon.')

    trips, stations = load_trips(G, input_csv, polygon=polygon)
    print(f'{save_name}: stations={len(stations)}, '
          f'trips={sum(trips.values())}')

    # Saving data
    ox.save_graphml(G, filepath=f'{output_folder}{save_name}.graphml')
    demand = h5py.File(f'{output_folder}{save_name}_demand.hdf5',
                       'w')
    demand.attrs['city'] = city_name
    demand.attrs['nbr of stations'] = len(stations)
    demand.attrs['nbr of trips'] = sum(trips.values())
    for k, v in trips.items():
        grp = demand.require_group(f'{k[0]}')
        grp[f'{k[1]}'] = v
    demand.close()


def gen_hom_demand(csv_path, poly_path, station_nbr, graph_path,
                   method="regular", station_buffer=3):
    """
    This function generates surrogate stations and an uniform demand of 1
    between those stations for the given polygon and graph. For generating the
    stations the st_sample function of the R package r-spatial is used.

    :param csv_path: Path where the csv file with the generated demand
    should be stored.
    :type csv_path: str
    :param poly_path: Path to the JSON of the area polygon.
    :type poly_path: str
    :param station_nbr: Number of stations to be generated
    :type station_nbr: int
    :param graph_path: Path to the graph of the area.
    :type graph_path: str
    :param method: Sampling method for the stations. 'random', 'hexagonal'
    (triangular really), 'regular'
    :type method: str
    :param station_buffer: Buffer for station number, if 'hexagonal'
    or 'regular' are chosen.
    :return: None
    """
    r_gen_hom_stations = r('''
        library(sf)
        gen_hom_stations <- function (poly_path, station_nbr, 
        method="regular", station_buffer=5)
        {
        # This function generates the stations for the homogenised demand.

        poly <- st_read(poly_path)
        # Transforming polygon into pseudo-mercator coordinates for sampling
        poly_transf <- st_transform(poly, 3857)
        # Sample the desired amount of stations
        if (method != "random") {
            # For non random distribution the number of stations to sample is 
            # increased, to ensure at least the amount of desired stations.
            samples_transf <- st_sample(poly_transf, 
                                        station_nbr+station_buffer, 
                                        type=method)
        } else {
            samples_transf <- st_sample(poly_transf, station_nbr, type=method)
        }

        # Transform the sampled stations back to WGS84
        samples <- st_transform(samples_transf,4326)

        # Print number of actually sampled stations and save as csv
        print(sprintf("Number of Stations: %s", length(samples)))
        df <- data.frame(matrix(unlist(samples), nrow=length(samples), 
                                byrow=TRUE))
        names(df)[1] <- "lon"
        names(df)[2] <- "lat"
        return(df)
        }
        ''')
    pandas2ri.activate()
    hom_stations = r_gen_hom_stations(poly_path=poly_path,
                                      station_nbr=station_nbr, method=method,
                                      station_buffer=station_buffer)

    hom_stations = {s_id + 1: (hom_stations['lat'][s_id],
                               hom_stations['lon'][s_id])
                    for s_id in range(len(hom_stations))}

    poly = get_polygon_from_json(poly_path)
    hom_stations = stations_in_polygon(hom_stations, poly)
    if len(hom_stations.keys()) > station_nbr:
        print('More stations than needed, removing excess stations.')
        g = ox.load_graphml(filepath=graph_path)
        new_stations = remove_stations(hom_stations, g, station_nbr)
    elif len(hom_stations.keys()) < station_nbr:
        print('Fewer stations than needed, please restart')
        new_stations = hom_stations
    else:
        print('Exact number stations as needed.')
        new_stations = hom_stations

    start_station = []
    start_location = []
    start_latitude = []
    start_longitude = []
    end_station = []
    end_location = []
    end_latitude = []
    end_longitude = []
    number_of_trips = []

    demand = {od: 1 for od in it.product(new_stations.keys(), repeat=2)
              if od[0] != od[1]}

    for od, t in demand.items():
        start_station.append(od[0])
        start_location.append(new_stations[od[0]])
        start_latitude.append(new_stations[od[0]][0])
        start_longitude.append(new_stations[od[0]][1])
        end_station.append(od[1])
        end_location.append(new_stations[od[1]])
        end_latitude.append(new_stations[od[1]][0])
        end_longitude.append(new_stations[od[1]][1])
        number_of_trips.append(t)

    data = [start_station, start_location, start_latitude, start_longitude,
            end_station, end_location, end_latitude, end_longitude,
            number_of_trips]

    columns = ['start station', 'start location', 'start latitude',
               'start longitude',
               'end station', 'end location', 'end latitude', 'end longitude',
               'number of trips']

    df = pd.DataFrame(data=data)
    df = df.transpose()
    df.columns = columns
    write_csv(df, csv_path)

    df.close()
