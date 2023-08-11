import sys

sys.path.append("..")

from setup_params import params
from setup_paths import paths
from src.plot import plot_city
import warnings


def main():
    warnings.filterwarnings("ignore")

    save = "hh"
    city = "Hamburg"

    # Number of different cylcists types. You have to adapt the vecotrs of speed and penalties to fit the number of cyclist types.
    # Should match the number of different cyclist types in the prep script.
    params["cyclist_types"] = 1

    # Speed of the cylcists (in m/s).
    params["speeds"] = [1]

    # Reduce the speed by the penalty factor.
    params["car_penalty"] = {
        "primary": [1 / 7],
        "secondary": [1 / 2.4],
        "tertiary": [1 / 1.4],
        "residential": [1 / 1.1],
    }
    params["slope_penalty"] = {
        0.06: [1 / 3.2],
        0.04: [1 / 2.2],
        0.02: [1 / 1.4],
        0.0: [1 / 1.0],
    }
    
    # Add a constant time penalty to the traveltime along an edge.
    params["intersection_penalty"] = {"large": [25], "medium": [15], "small": [5]}
    params["turn_penalty"] = {"large": [20], "medium": [10], "small": [0]}
    params["bp_end_penalty"] = {
        "primary": {1: [10], 2: [15], 3: [25]},
        "secondary": {1: [5], 2: [10], 3: [15]},
        "tertiary": {1: [0], 2: [5], 3: [7.5]},
        "residential": {1: [0], 2: [1], 3: [1]},
    }


    # Setings form plots
    params["legends"] = False
    params["titles"] = False
    params["plot_format"] = "png"

    params["plot_bp_comp"] = True

    params["plot_evo"] = False
    params["evo_for"] = [(False, 1, False)]

    # Usefull to set to true, if nothing changed in the comp. state calculation.
    params["cached_comp_state"] = False

    modes = [
        (False, 1, False),
        (False, 1, True),
    ]

    plot_city(city, save, modes=modes, paths=paths, params=params)


if __name__ == "__main__":
    main()
