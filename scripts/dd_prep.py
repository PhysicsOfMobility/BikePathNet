import json
import sys
from ast import literal_eval
from os.path import join
from setup_paths import paths
from shutil import copyfile

sys.path.append(paths["project_dir"])
from src.prep_and_plot import gen_noisy_demand

save = "dd"
delta_noise = 0.20

with open(join(paths["input_folder"], save, f"{save}_demand.json"), "r") as f:
    demand = json.load(f)

for i in range(1, 11):
    print(f"Generating data set {i}")
    with open(join(paths["input_folder"], f"{save}_noisy_demand_{i}", f"{save}_noisy_demand_{i}_node_noise.json"), "r") as f:
        nodes_noise = json.load(f)
    nodes_noise = {literal_eval(k): v for k, v in nodes_noise.items()}
    gen_noisy_demand(demand, delta_noise, save, f"{save}_noisy_demand_{i}", nodes_noise=nodes_noise)
    for pen in ["car_penalty", "slope_penalty", "intersection_penalty", "turn_penalty"]:
        noisy_penalty_save = f"{save}_noisy_{pen}_{i}"
        copyfile(join(paths["input_folder"], save, f"{save}.graphml"),
             join(paths["input_folder"], noisy_penalty_save, f"{noisy_penalty_save}.graphml"))
        copyfile(join(paths["input_folder"], save, f"{save}_demand.json"),
             join(paths["input_folder"], noisy_penalty_save, f"{noisy_penalty_save}_demand.json"))