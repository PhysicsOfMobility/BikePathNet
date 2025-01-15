import sys
from os.path import join
from setup_paths import paths
from setup_params import params
sys.path.append(paths["project_dir"])
from src.prep_and_plot import plot_city

city_save = "hh"
city_name = "Hamburg"

paths["polygon_file"] = join(paths["polygon_folder"], f"{city_save}.json")
paths["graph_file"] = join(paths["input_folder"], city_save, f"{city_save}.graphml")
paths["demand_file"] = join(paths["input_folder"], city_save, f"{city_save}_demand.json")

params["stat_usage_norm"] = 881 / 365

params["plot_evo"] = False
params["evo_for"] = [(False, 1, False)]

modes = [
    (False, 1, False),
    (False, 1, True),
]

plot_city(city_name, city_save, modes=modes, paths=paths, params=params)
