import json
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely as shp
from pathlib import Path
from os.path import join
from scipy import optimize
from shapely import centroid
from shapely.geometry import Point, Polygon, MultiPolygon
from data_helper import get_street_type, get_polygon_from_json, save_polygon_as_json


def save_data(
    demand: pd.DataFrame,
    polygon: Polygon,
    g: nx.MultiDiGraph | nx.MultiGraph,
    od_matrix: np.array,
    od_distances: np.array,
    node_mapping: dict,
    len_scale_all: float,
    len_scale_bike: float,
    modal_share_bike: float,
    p_bike: float,
    save_name: str,
    paths: dict,
):
    """
    Save generated graph, demand and relevant metadata.
    Parameters
    ----------
    demand: pd.DataFrame
    polygon: Polygon
    g: nx.MultiDiGraph | nx.MultiGraph
    od_matrix: np.array
    od_distances: np.array
    node_mapping: dict
    len_scale_all: float
    len_scale_bike: float
    modal_share_bike: float
    p_bike: float
    save_name: str
    paths: dict

    Returns
    -------

    """
    csv_folder = join(paths["data_dir"], "csvs")
    Path(csv_folder).mkdir(parents=True, exist_ok=True)
    demand.to_csv(join(paths["data_dir"], "csvs", f"{save_name}.csv"), index=False)

    polygon_folder = join(paths["data_dir"], "polygons")
    Path(polygon_folder).mkdir(parents=True, exist_ok=True)
    save_polygon_as_json(polygon, join(polygon_folder, f"{save_name}.json"))

    raw_folder = join(paths["data_dir"], "raw", save_name)
    Path(raw_folder).mkdir(parents=True, exist_ok=True)
    ox.save_graphml(g, join(raw_folder, f"{save_name}_buffered.graphml"))

    np.savetxt(join(raw_folder, f"{save_name}_od_matrix.csv"), od_matrix, delimiter=",")
    np.savetxt(
        join(raw_folder, f"{save_name}_od_distances.csv"), od_distances, delimiter=","
    )

    with open(join(raw_folder, f"{save_name}_node_mapping.json"), "w") as fp:
        json.dump(node_mapping, fp)

    parameters = {
        "node mapping": node_mapping,
        "length scale general trips": len_scale_all,
        "length scale bike trips": len_scale_bike,
        "modal share bikes": modal_share_bike,
        "p tilde bike": p_bike,
    }
    with open(join(raw_folder, f"{save_name}_params.json"), "w") as fp:
        json.dump(parameters, fp)


def general_demand_single_origin(
    L, L_scale: float = 1.0, Z_O: float = 1.0, N_D: float = 1.0
):
    return N_D / Z_O * np.exp(-L / L_scale)


def general_demand_single_origin_norm(
    O, Ds, OD_distances, Z_O, Ns, L_scale: float = 1.0
):
    return sum(
        [
            general_demand_single_origin(
                L=OD_distances[O][D], L_scale=L_scale, Z_O=Z_O, N_D=Ns[D]
            )
            for D in Ds
        ]
    )


def find_general_demand_single_origin_Z_O(
    O, Ds, OD_distances, Ns, L_scale: float = 1.0, abs_norm: float = 1.0
):
    opt_func = lambda z: abs(
        general_demand_single_origin_norm(O, Ds, OD_distances, z, Ns, L_scale)
        - abs_norm
    )
    return optimize.minimize(opt_func, np.array(1.0), method="Nelder-Mead", tol=1e-6).x[
        0
    ]


def generate_demand_Z_Os(
    od_stations, od_distances, Ns, L_scale: float = 1.0, round_trips: bool = True
):
    norms = np.zeros(len(od_stations))
    for origin in od_stations:
        if round_trips:
            destination_nodes = od_stations
        else:
            destination_nodes = set(od_stations) - {origin}
        norms[origin] = find_general_demand_single_origin_Z_O(
            origin, destination_nodes, od_distances, Ns, L_scale=L_scale
        )
    return norms


def f(L, L_scale):
    return (L / L_scale) ** 2 * np.exp(-L / L_scale)


def bike_prob(p, L, L_scale):
    return 1 - np.exp(-p * f(L, L_scale))


def get_modal_share(
    od_stations: list,
    od_distances: np.array,
    od_matrix: np.array,
    Ns,
    p: float = 0.5,
    L_mode_scale: float = 1.0,
    mode_prob=bike_prob,
):
    norm_factor = 1 / sum(Ns)
    share = sum(
        od_matrix[o, d] * mode_prob(p=p, L=od_distances[o, d], L_scale=L_mode_scale)
        for d in od_stations
        for o in od_stations
    )

    return norm_factor * share


def calibrate_p(
    modal_share_goal,
    od_stations,
    od_distances,
    od_matrix,
    Ns,
    L_mode_scale=1.0,
    mode_prob=bike_prob,
):
    opt_func = lambda p: abs(
        get_modal_share(
            od_stations=od_stations,
            od_distances=od_distances,
            od_matrix=od_matrix,
            Ns=Ns,
            p=p,
            L_mode_scale=L_mode_scale,
            mode_prob=mode_prob,
        )
        - modal_share_goal
    )
    return optimize.minimize(
        opt_func, np.array(modal_share_goal), method="Nelder-Mead", tol=1e-6
    ).x[0]


def generate_OD_matrix(
    od_stations: list,
    od_distances: np.array,
    Ns: list,
    L_scale: float = 1.0,
    round_trips: bool = True,
):
    """

    Parameters
    ----------
    od_stations: list
    od_distances: np.array
    Ns: list
    L_scale: float
    round_trips: bool

    Returns
    -------

    """
    Z_Os = generate_demand_Z_Os(
        od_stations=od_stations,
        od_distances=od_distances,
        Ns=Ns,
        L_scale=L_scale,
        round_trips=round_trips,
    )

    od_matrix = np.zeros((len(od_stations), len(od_stations)))
    for o in od_stations:
        for d in od_stations:
            od_matrix[o, d] = Ns[o] * general_demand_single_origin(
                L=od_distances[o, d], L_scale=L_scale, Z_O=Z_Os[o], N_D=Ns[d]
            )

    if not round_trips:
        np.fill_diagonal(od_matrix, 0.0)

    return od_matrix


def generate_mode_OD_matrix(
    od_stations: list,
    od_distances: np.array,
    od_matrix: np.array,
    Ns,
    modal_share_goal: float,
    L_mode_scale: float = 1.0,
    mode_prob=bike_prob,
):
    """

    Parameters
    ----------
    od_stations: list
    od_distances: np.array
    od_matrix: np.array
    Ns
    modal_share_goal: float
    L_mode_scale: float
    mode_prob=bike_prob

    Returns
    -------

    """
    p = calibrate_p(
        modal_share_goal,
        od_stations,
        od_distances,
        od_matrix,
        Ns,
        L_mode_scale=L_mode_scale,
        mode_prob=mode_prob,
    )

    mode_od_matrix = np.zeros((len(od_stations), len(od_stations)))
    for o in od_stations:
        for d in od_stations:
            mode_od_matrix[o, d] = od_matrix[o, d] * mode_prob(
                p=p, L=od_distances[o, d], L_scale=L_mode_scale
            )

    return mode_od_matrix, p


def get_od_shortest_paths_and_distances(
    od_stations: list, g: nx.MultiDiGraph | nx.DiGraph, od_node_mapping: dict
):
    """
    Return shortest paths and distances between all od stations in graph 'g'.
    Parameters
    ----------
    od_stations: list
    g: nx.MultiDiGraph | nx.MultiGraph
    od_node_mapping: dict

    Returns
    -------

    """
    od_distances = np.zeros((len(od_stations), len(od_stations)))
    od_shortest_paths = np.empty((len(od_stations), len(od_stations)), dtype=np.ndarray)

    for o in od_stations:
        o_mapped = od_node_mapping[o]
        lengths, paths = nx.single_source_dijkstra(g, o_mapped, weight="length")
        for d in od_stations:
            od_distances[o, d] = lengths[od_node_mapping[d]]
            od_shortest_paths[o, d] = paths[od_node_mapping[d]]

    return od_distances, od_shortest_paths


def get_od_stations(
    pop_dens: pd.DataFrame, g: nx.MultiDiGraph | nx.MultiGraph, cutoff: float = 400
):
    """
    Generate od stations based on population density 'pop_dens' and graph 'g'. If the centre of the population density
    cell are further away than 'cutoff', then this cell is ignored.

    Parameters
    ----------
    pop_dens: pd.DataFrame
    g: nx.MultiDiGraph | nx.MultiGraph
    cutoff: float

    Returns
    -------

    """
    stations = pop_dens[["population", "centroid", "geometry"]]
    nearest_node, dist_to_nn = ox.nearest_nodes(
        g, pop_dens["centroid"].x, pop_dens["centroid"].y, return_dist=True
    )
    stations["nearest_node"] = nearest_node
    stations["dist_to_nn"] = dist_to_nn

    stations = stations[stations.dist_to_nn <= cutoff]
    stations.reset_index(drop=True, inplace=True)

    stations.rename_axis("station", inplace=True)
    stations.reset_index(inplace=True)

    stations["in_poly"] = [g.nodes[n]["in_poly"] for n in stations["nearest_node"]]
    stations.rename(columns={"centroid": "location"}, inplace=True)

    node_mapping_g = pd.Series(
        stations.nearest_node.values, index=stations.station
    ).to_dict()
    od_stations = stations["station"].to_numpy()

    return od_stations, stations, node_mapping_g


def hex_split_six(polygon: Polygon):
    """
    Split a hexagonal polygon into six equal parts.

    Parameters
    ----------
    polygon: Polygon

    Returns
    -------
    output : Polygon
    """
    poly_coords = list(polygon.exterior.coords)
    return MultiPolygon(
        [
            Polygon(
                [poly_coords[i], centroid(polygon), poly_coords[i + 1], poly_coords[i]]
            )
            for i in range(6)
        ]
    )


def hex_split_three(polygon: Polygon):
    """
    Split a hexagonal polygon into three equal parts.

    Parameters
    ----------
    polygon: Polygon

    Returns
    -------
    output : Polygon
    """
    poly_coords = list(polygon.exterior.coords)
    return MultiPolygon(
        [
            Polygon(
                [
                    poly_coords[i],
                    centroid(polygon),
                    poly_coords[i + 2],
                    poly_coords[i + 1],
                    poly_coords[i],
                ]
            )
            for i in range(0, 6, 2)
        ]
    )


def hex_split_pop_density(
    pop_density: pd.DataFrame,
    local_crs: int,
    global_crs: int = 3857,
    split_size: int = 6,
):
    """

    Parameters
    ----------
    pop_density: pd.DataFrame
    local_crs: int
    global_crs: int
    split_size: int

    Returns
    -------

    """
    if split_size == 3:
        split_function = hex_split_three
    elif split_size == 6:
        split_function = hex_split_six
    else:
        print("split_size must be either 3 or 6")
        return -1
    pop_density.to_crs(local_crs, inplace=True)

    pop_density_split = pop_density[["h3", "population", "geometry"]]
    pop_density_split.loc[:, "population"] = (
        pop_density_split.loc[:, "population"].div(split_size).round(2)
    )
    pop_density_split.loc[:, "geometry"] = pop_density.loc[:, "geometry"].apply(
        lambda p: split_function(p)
    )
    pop_density_split = pop_density_split.explode(index_parts=False, ignore_index=False)
    pop_density_split.set_geometry("geometry", crs=local_crs, inplace=True)

    pop_density_split.to_crs(global_crs, inplace=True)
    return pop_density_split


def find_first_point_in_polygon(
    origin: int,
    destination: int,
    polygon: Polygon,
    stations: pd.DataFrame,
    sps: np.ndarray,
    g: nx.MultiDiGraph | nx.MultiGraph,
):
    """

    Parameters
    ----------
    origin: int
    destination: int
    polygon: Polygon
    stations: pd.DataFrame
    sps: np.ndarray
    g: nx.MultiDiGraph | nx.MultiGraph

    Returns
    -------

    """
    start_point = stations[stations["station"] == origin]["location"].values[0]
    if polygon.contains(start_point):
        return start_point, True
    else:
        sp = sps[origin, destination]
        for n in sp:
            if g.nodes[n]["in_poly"]:
                return Point(g.nodes[n]["x"], g.nodes[n]["y"]), False
    return None, False


def find_last_point_in_polygon(
    origin: int,
    destination: int,
    polygon: Polygon,
    stations: pd.DataFrame,
    sps: np.ndarray,
    g: nx.MultiDiGraph | nx.MultiGraph,
):
    """

    Parameters
    ----------
    origin: int
    destination: int
    polygon: Polygon
    stations: pd.DataFrame
    sps: np.ndarray
    g: nx.MultiDiGraph | nx.MultiGraph

    Returns
    -------

    """
    end_point = stations[stations["station"] == destination]["location"].values[0]
    if polygon.contains(end_point):
        return end_point, True
    else:
        sp = sps[origin, destination]
        node_in_poly = [g.nodes[n]["in_poly"] for n in sp]
        try:
            end_node = sp[
                next(i for i in reversed(range(len(node_in_poly))) if node_in_poly[i])
            ]
            return Point(g.nodes[end_node]["x"], g.nodes[end_node]["y"]), False
        except StopIteration:
            return None, False


def create_demand(
    pop_dist_file: str,
    polygon_file: str,
    graph_file: str = None,
    modal_share: float = 0.2,
    length_scale_general: float = 10000.0,
    length_scale_bike: float = 2500.0,
    round_trips: bool = True,
    buffer_dist: float = 1000.0,
    crs_glob: int = 4326,
    crs_loc: int = 32633,
    node_dist_cutoff: float = 400.0,
    trip_cutoff: float = 0.0,
    save_name: str | None = None,
    paths: dict | None = None,
    nw: str = "drive",
    intersection_tol: float = 10.0,
    street_types_to_remove: list[str] | None = None,
):
    """

    Parameters
    ----------
    pop_dist_file: str
    polygon_file: str
    graph_file: str
    modal_share: float
    length_scale_general: float
    length_scale_bike: float
    round_trips: bool
    buffer_dist: float
    crs_glob: int
    crs_loc: int
    node_dist_cutoff: float
    trip_cutoff: float
    save_name: str | None
    paths: dict | None
    nw: str
    intersection_tol: float
    street_types_to_remove: list[str] | None

    Returns
    -------

    """
    if street_types_to_remove is None:
        street_types_to_remove = ["motorway", "motorway_link", "trunk", "trunk_link"]
    if paths is None:
        paths = {}

    pop_dist = gpd.read_file(pop_dist_file)
    polygon = get_polygon_from_json(polygon_file)

    pop_dist.to_crs(crs_loc, inplace=True)
    pop_dist["centroid"] = pop_dist.centroid

    polygon_projected = gpd.GeoDataFrame(
        {"geometry": [polygon]}, geometry="geometry", crs=crs_glob
    )
    polygon_projected = polygon_projected.to_crs(crs_loc)["geometry"][0]

    poly_buffered_proj = shp.buffer(polygon_projected, buffer_dist)
    poly_buffered = gpd.GeoDataFrame(
        {"geometry": [poly_buffered_proj]}, geometry="geometry", crs=crs_loc
    )
    poly_buffered = poly_buffered.to_crs(crs_glob)["geometry"][0]

    if graph_file is None:
        print("Downloading and preparing graph.")
        g = ox.graph_from_polygon(poly_buffered, network_type=nw)
        edges_to_remove = [
            e
            for e in g.edges(keys=True)
            if get_street_type(g, e, motorway=street_types_to_remove)
            in street_types_to_remove
        ]
        g.remove_edges_from(edges_to_remove)
        h = ox.project_graph(g, to_crs=f"epsg:{crs_loc}")
        h = ox.consolidate_intersections(
            h,
            tolerance=intersection_tol,
            rebuild_graph=True,
            dead_ends=True,
            reconnect_edges=True,
        )
        h = nx.convert_node_labels_to_integers(h)
        h = ox.truncate.largest_component(h, strongly=True)
        nx.set_node_attributes(h, {n: n for n in h.nodes}, "osmid")
    else:
        g = ox.load_graphml(graph_file)
        h = ox.project_graph(g, to_crs=f"epsg:{crs_loc}")

    nx.set_node_attributes(
        h,
        {
            n: polygon_projected.contains(Point((d["x"], d["y"])))
            for n, d in h.nodes(data=True)
        },
        name="in_poly",
    )
    g = ox.project_graph(h, to_crs=f"epsg:{crs_glob}")

    print("Reducing population data to local data.")
    pop_dist.to_crs(crs_loc, inplace=True)
    pop_dist["in_buffered_poly"] = poly_buffered_proj.contains(pop_dist.centroid)
    local_data = pop_dist.loc[
        pop_dist["in_buffered_poly"], ["population", "geometry", "centroid"]
    ]
    local_data.reset_index(drop=True, inplace=True)
    local_data["in_poly"] = polygon_projected.contains(local_data.centroid)
    print(
        f"Local population data with {len(local_data)} cells, {len(local_data[local_data['in_poly']])} within the observed area and {len(local_data[~local_data['in_poly']])} outside."
    )

    print(f"Getting od nodes for cells, with distance cutoff {node_dist_cutoff}m.")
    od_stations, stations, node_mapping = get_od_stations(
        local_data, h, cutoff=node_dist_cutoff
    )
    od_distances, od_shortest_paths = get_od_shortest_paths_and_distances(
        od_stations, g, node_mapping
    )
    node_sizes = stations["population"].values
    od_matrix = generate_OD_matrix(
        od_stations=od_stations,
        od_distances=od_distances,
        Ns=node_sizes,
        L_scale=length_scale_general,
        round_trips=round_trips,
    )
    bike_od_matrix, p_bike = generate_mode_OD_matrix(
        od_stations=od_stations,
        od_distances=od_distances,
        od_matrix=od_matrix,
        Ns=node_sizes,
        modal_share_goal=modal_share,
        L_mode_scale=length_scale_bike,
    )
    print(f"Discarded {len(local_data)-len(stations)} cells due to distance cutoff.")

    demand = pd.merge(
        stations[["location", "station"]],
        stations[["location", "station"]],
        how="cross",
    )
    demand = demand.assign(trips=lambda s: bike_od_matrix[s.station_x, s.station_y])
    demand.drop(demand[demand.trips == 0].index, inplace=True)
    demand.drop(demand[demand.trips < trip_cutoff].index, inplace=True)
    demand.rename(
        columns={
            "location_x": "startpoint",
            "station_x": "start station",
            "location_y": "endpoint",
            "station_y": "end station",
            "trips": "number of trips",
        },
        inplace=True,
    )
    demand.reset_index(inplace=True, drop=True)
    print(
        f"{len(demand)} unique trips in the buffered area between {len(stations)} od nodes."
    )

    fp = []
    lp = []
    inside_trips = 0
    for index, row in demand.iterrows():
        fp_idx, fp_inside = find_first_point_in_polygon(
            row["start station"],
            row["end station"],
            polygon_projected,
            stations,
            od_shortest_paths,
            h,
        )
        lp_idx, lp_inside = find_last_point_in_polygon(
            row["start station"],
            row["end station"],
            polygon_projected,
            stations,
            od_shortest_paths,
            h,
        )
        fp.append(fp_idx)
        lp.append(lp_idx)
        if fp_inside and lp_inside:
            inside_trips += 1
    print(f"{inside_trips} trips start and end inside the observed area.")

    trips_before = len(demand)
    demand["firstpoint"] = gpd.GeoSeries(fp, crs=crs_loc).to_crs(crs_glob)
    demand["lastpoint"] = gpd.GeoSeries(lp, crs=crs_loc).to_crs(crs_glob)
    demand.dropna(
        inplace=True
    )  # Drop demand where one nearest node is outside the unbuffered polygon.
    trips_after = len(demand)
    print(f"Dropping {trips_before-trips_after} trips not crossing the observed area.")

    demand["start longitude"] = demand["firstpoint"].apply(lambda p: p.x)
    demand["start latitude"] = demand["firstpoint"].apply(lambda p: p.y)
    demand["end longitude"] = demand["lastpoint"].apply(lambda p: p.x)
    demand["end latitude"] = demand["lastpoint"].apply(lambda p: p.y)

    demand["start location"] = demand["firstpoint"].apply(lambda p: (p.y, p.x))
    demand["end location"] = demand["lastpoint"].apply(lambda p: (p.y, p.x))

    demand.drop(
        columns=["startpoint", "firstpoint", "endpoint", "lastpoint"], inplace=True
    )
    demand.reset_index(inplace=True, drop=True)
    print(
        f"{len(demand)} unique trips, with {demand['number of trips'].sum():3.2f} travellers, crossing the observed area."
    )

    stations.set_geometry("location", inplace=True)

    if save_name is not None:
        save_data(
            demand,
            polygon,
            g,
            od_matrix,
            od_distances,
            node_mapping,
            length_scale_general,
            length_scale_bike,
            modal_share,
            p_bike,
            save_name,
            paths,
        )

    return (
        demand,
        polygon,
        g,
        od_matrix,
        od_distances,
        od_shortest_paths,
        stations,
        node_mapping,
        length_scale_general,
        length_scale_bike,
        modal_share,
        p_bike,
    )
