"""
The core algorithm of the bikeability optimisation project.
"""
import h5py
import json
from os.path import join
from pathlib import Path
from .setup_helper import create_default_params, create_default_paths
from julia import Main


def run_simulation(city, save, params=None, paths=None, mode=(False, 1, False)):
    """
    Prepares everything to run the core algorithm. All data will be saved to
    the given folders.
    :param city: Name of the city.
    :type city: str
    :param save: save name of everything associated with de place.
    :type save: str
    :param mode: mode of the algorithm. (buildup, minmode, ex_inf)
    :type mode: tuple
    :param paths: Dictionary with folder paths (see example folder)
    :type paths: dict or None
    :param params: Dictionary with parameters (see example folder)
    :type params: dict or None
    :return: None
    """

    params_file, input_folder, output_folder, log_folder = create_algorithm_params_file(
        save, params, paths, mode
    )
    Main.eval("using DrWatson")
    Main.eval('@quickactivate "BikePathNet"')

    Main.eval('include(srcdir("node.jl"))')
    Main.eval('include(srcdir("edge.jl"))')
    Main.eval('include(srcdir("graph.jl"))')
    Main.eval('include(srcdir("shortestpath.jl"))')
    Main.eval('include(srcdir("trip.jl"))')
    Main.eval('include(srcdir("logging_helper.jl"))')
    Main.eval('include(srcdir("algorithm_helper.jl"))')
    Main.eval('include(srcdir("algorithm.jl"))')

    Main.city = city
    Main.save = save

    Main.input_folder = input_folder
    Main.output_folder = output_folder
    Main.log_folder = log_folder
    Main.params_file = params_file
    Main.mode = mode

    Main.eval(
        "run_simulation(city, save, input_folder, "
        "output_folder, params_file, log_folder)"
    )

    return 0


def create_algorithm_params_file(save, params=None, paths=None, mode=(False, 1, False)):
    """
    Prepares everything to run the core algorithm.
    :param save: save name of everything associated with de place.
    :type save: str
    :param mode: mode of the algorithm. (buildup, minmode, ex_inf)
    :type mode: tuple
    :param paths: Dictionary with folder paths (see example folder)
    :type paths: dict or None
    :param params: Dictionary with parameters (see example folder)
    :type params: dict or None
    :return: Location of params file
    """
    if paths is None:
        paths = create_default_paths()
    if params is None:
        params = create_default_params()

    input_folder = join(paths["input_folder"], save)
    output_folder = join(paths["output_folder"], save)
    log_folder = join(paths["log_folder"], save)
    # Check if necessary folders exists, otherwise create.
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(log_folder).mkdir(parents=True, exist_ok=True)

    if params["car_penalty"] is None:
        params["car_penalty"] = {k: 1.0 for k in params["turn_penalty"].keys()}
    if params["slope_penalty"] is None:
        params["slope_penalty"] = dict()
    if params["surface_penalty"] is None:
        params["surface_penalty"] = dict()
    if params["intersection_penalty"] is None:
        params["intersection_penalty"] = {"large": 1.0, "medium": 1.0, "small": 1.0}
    if params["turn_penalty"] is None:
        params["turn_penalty"] = {"large": 1.0, "medium": 1.0, "small": 1.0}

    algorithm_params = dict()
    algorithm_params["mode"] = mode
    algorithm_params["save_edge_load"] = params["save_edge_load"]
    algorithm_params["cyclist_types"] = params["cyclist_types"]
    algorithm_params["speeds"] = params["speeds"]
    algorithm_params["car_penalty"] = params["car_penalty"]
    algorithm_params["slope_penalty"] = params["slope_penalty"]
    algorithm_params["surface_penalty"] = params["surface_penalty"]
    algorithm_params["intersection_penalty"] = params["intersection_penalty"]
    algorithm_params["turn_penalty"] = params["turn_penalty"]
    algorithm_params["bp_end_penalty"] = params["bp_end_penalty"]

    params_file = join(
        input_folder, f"{save}_algorithm_params_{mode[0]:d}{mode[1]}{mode[2]:d}.json"
    )
    with open(params_file, "w") as fp:
        json.dump(algorithm_params, fp)

    return params_file, input_folder, output_folder, log_folder


def calc_comparison_state(
    save,
    state,
    bike_paths=None,
    ex_inf=False,
    blocked=False,
    use_penalties=True,
    base_state=False,
    opt_state=False,
    params=None,
    paths=None,
    use_base_folder=False,
):
    """
    Calculates the data for a comparison bike path network. If no bike paths
    are given, the trips are calculated in a bike path free graph.
    :param save: save name of everything associated with de place.
    :type save: str
    :param state: Name of the comparison state
    :type state: str
    :param bike_paths: List of edges which have a bike path.
    :type bike_paths: list
    :param ex_inf: If existing infrastructure should be considered
    :type: ex_inf: bool
    :param use_penalties: If penalties should be used for sp calculation
    :type use_penalties: bool
    :param base_state: Base state calculation
    :type base_state: bool
    :param opt_state: Optimal state calculation
    :type opt_state: bool
    :param params: Dict with the params for plots etc., check 'params.py' in
    the example folder.
    :type params: dict
    :param paths: Dict with the paths for plots etc.
    :type paths: dict
    :return: Data structured as from the main algorithm.
    :rtype: dict
    """
    if params is None:
        params = create_default_params()
    if paths is None:
        paths = create_default_paths()
    if bike_paths is None:
        bike_paths = []
    if base_state:
        use_penalties = True
        ex_inf = False
    if opt_state:
        use_penalties = False
        ex_inf = False

    if use_base_folder:
        input_folder = join(paths["input_folder"], save.split(paths["save_devider"])[0])
        output_folder = join(
            paths["output_folder"], save.split(paths["save_devider"])[0]
        )
    else:
        input_folder = join(paths["input_folder"], save)
        output_folder = join(paths["output_folder"], save)

    comp_state_params = dict()
    comp_state_params["bike_paths"] = bike_paths
    comp_state_params["ex_inf"] = ex_inf
    comp_state_params["blocked"] = blocked
    comp_state_params["cyclist_types"] = params["cyclist_types"]
    comp_state_params["speeds"] = params["speeds"]
    if use_penalties:
        # Set penalties for different street types
        comp_state_params["car_penalty"] = params["car_penalty"]
        comp_state_params["slope_penalty"] = params["slope_penalty"]
        comp_state_params["surface_penalty"] = params["surface_penalty"]
        comp_state_params["intersection_penalty"] = params["intersection_penalty"]
        comp_state_params["turn_penalty"] = params["turn_penalty"]
        comp_state_params["bp_end_penalty"] = params["bp_end_penalty"]
    else:
        comp_state_params["car_penalty"] = {
            k: [1] * params["cyclist_types"] for k in params["car_penalty"].keys()
        }
        comp_state_params["slope_penalty"] = {
            k: [1] * params["cyclist_types"] for k in params["slope_penalty"].keys()
        }
        comp_state_params["surface_penalty"] = {
            k: [1] * params["cyclist_types"] for k in params["surface_penalty"].keys()
        }
        comp_state_params["intersection_penalty"] = {
            k: [0] * params["cyclist_types"]
            for k in params["intersection_penalty"].keys()
        }
        comp_state_params["turn_penalty"] = {
            k: [0] * params["cyclist_types"] for k in params["turn_penalty"].keys()
        }
        comp_state_params["bp_end_penalty"] = {
            k: {
                1: [0] * params["cyclist_types"],
                2: [0] * params["cyclist_types"],
                3: [0] * params["cyclist_types"],
            }
            for k in params["bp_end_penalty"].keys()
        }

    params_file = join(input_folder, f"{save}_comparison_state_params_{state}.json")
    with open(params_file, "w") as fp:
        json.dump(comp_state_params, fp)

    if not params["cached_comp_state"]:
        Main.eval("using DrWatson")
        Main.eval('@quickactivate "BikePathNet"')

        Main.save = save
        Main.state = state

        Main.input_folder = input_folder
        Main.output_folder = output_folder
        Main.params_file = params_file

        Main.use_base_folder = use_base_folder

        Main.eval('include(srcdir("node.jl"))')
        Main.eval('include(srcdir("edge.jl"))')
        Main.eval('include(srcdir("graph.jl"))')
        Main.eval('include(srcdir("shortestpath.jl"))')
        Main.eval('include(srcdir("trip.jl"))')
        Main.eval('include(srcdir("logging_helper.jl"))')
        Main.eval('include(srcdir("algorithm_helper.jl"))')
        Main.eval('include(srcdir("algorithm.jl"))')

        Main.eval(
            "calc_comparison_state("
            "save, state, input_folder, output_folder, params_file, use_base_save_input=use_base_folder"
            ")"
        )

    data = dict()
    hf = h5py.File(
        join(output_folder, f"{save}_data_comparison_state_{state}.hdf5"),
        "r",
    )

    data["bike_paths"] = json.loads(hf["bike_paths"][()])
    data["bpp"] = hf["bpp"][()]
    data["cost"] = hf["cost"][()]
    data["trdt"] = json.loads(hf["trdt"][()])
    data["tfdt"] = hf["tfdt"][()]
    data["nos"] = hf["nos"][()]
    data["edge_load"] = json.loads(hf["edge_load"][()])

    return data
