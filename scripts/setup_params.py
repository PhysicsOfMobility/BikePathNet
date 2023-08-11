import numpy as np

params = {}
params["cyclist_types"] = 1
params["cyclist_split"] = {1: 1}
params["speeds"] = [1]
params["car_penalty"] = {
    "primary": [1 / 7],
    "secondary": [1 / 2.4],
    "tertiary": [1 / 1.4],
    "residential": [1 / 1.1],
}
params["slope_penalty"] = {0.06: [1.0], 0.04: [1.0], 0.02: [1.0], 0.00: [1.0]}
params["surface_penalty"] = {
    "asphalt": [1.0],
    "gravel": [1.0],
    "cobbel": [1.0],
}
params["turn_penalty"] = {
    "large": [0.0],
    "medium": [0.0],
    "small": [0.0],
}
params["intersection_penalty"] = {
    "large": [0.0],
    "medium": [0.0],
    "small": [0.0],
}
params["bp_end_penalty"] = {
    "primary": {1: [0.0], 2: [0.0], 3: [0.0]},
    "secondary": {1: [0.0], 2: [0.0], 3: [0.0]},
    "tertiary": {1: [0.0], 2: [0.0], 3: [0.0]},
    "residential": {1: [0.0], 2: [0.0], 3: [0.0]},
}

params["modes"] = [(1, False, 1)]  # Modes used for algorithm and plotting
params["save_edge_load"] = False
params["cut"] = True  # If results should be normalised to first removal
params["bike_paths"] = []  # Additional ike Paths for current state
params["use_exinf"] = False
params["ex inf"] = ["track", "lane"]  # Ex bp which should not be removed

params["prerun_comp_states"] = False  # Creat json for comp states
params["cached_comp_state"] = False  # Use existing comp state data if True

params["correct_area"] = True  # Correction of the area size

params["plot_bp_comp"] = True  # If BP comp should be plotted
params["plot_evo"] = False  # If BP evolution should be plotted
params["evo_for"] = []  # For given modes
params["bpp_range"] = [float(f"{i:4.3f}") for i in np.linspace(0, 1, num=101)]

params["dpi"] = 300  # dpi of plots
params["titles"] = True  # If figure title should be plotted
params["legends"] = True  # If legends should be plotted
params["plot_format"] = "png"  # Format of the saved plot

params["fs_title"] = 12  # Fontsize for title
params["fs_axl"] = 10  # Fontsize for axis labels
params["fs_ticks"] = 8  # Fontsize for axis tick numbers
params["fs_legend"] = 6  # Fontsize for legends

params["figs_ba_cost"] = (4, 3.5)  # Figsize ba-cost plot
params["c_ba"] = "#0080c0"  # Colour for ba in normal plot
params["lw_ba"] = 1
params["m_ba"] = "D"  # Marker for ba
params["ms_ba"] = 5  # Marker size for ba
params["c_cost"] = "#4d4d4d"  # Colour for cost in normal plot
params["lw_cost"] = 1
params["m_cost"] = "s"
params["ms_cost"] = 5  # Marker size for cost

params["figs_los_nos"] = (4, 3.5)  # Figsize nos-los plot
params["c_los"] = "#e6194B"  # Colour for nos in normal plot
params["lw_los"] = 1
params["m_los"] = "8"
params["ms_los"] = 5  # Marker size for los
params["c_nos"] = "#911eb4"
params["lw_nos"] = 1
params["m_nos"] = "v"
params["ms_nos"] = 5

params["c_st"] = {
    "primary": "#4d4d4d",
    "secondary": "#666666",
    "tertiary": "#808080",
    "residential": "#999999",
    "bike paths": "#0080c0",
}
params["figs_comp_st"] = (0.75, 1.85)  # Figsize for comp_st_driven plot

params["figs_snetwork"] = (7, 5)
params["ns_snetwork"] = 1
params["nc_snetwork"] = "#a9a9a9"
params["ew_snetwork"] = 0.6
params["ec_snetwork"] = "#b3b3b3"

params["figs_station_usage"] = (2.7, 2.6)
params["figs_station_usage_hist"] = (3.65, 1.3)
params["stat_usage_norm"] = 1
params["cmap_nodes"] = "cool"  # Cmap for nodes usage
params["nodesize"] = 7
params["ec_station_usage"] = "#b3b3b3"

params["figs_bp_evo"] = (2.2, 2.2)
params["lw_legend_bp_evo"] = 4

params["figs_bp_comp"] = (2.2, 2.2)
params["nc_bp_comp"] = "#d726ffff"
params["color_algo"] = "#007fbfff"
params["color_algo_ex_inf"] = "#000000"
params["color_cs"] = "#800000"
params["color_cs_ex_inf"] = "#000000"
params["color_both"] = "#f58231"
params["color_both_ex_inf"] = "#000000"
params["color_ex_inf"] = "#000000"
params["color_unused"] = "#b3b3b3"  # 808080

params["c_ed"] = "#0080c0"  # Colour ba for emp demand in rd-ed comp
params["c_rd"] = "#f58231"  # Colour ba for rand demand in rd-ed comp
params["lw_ed"] = 0.9
params["lw_rd"] = 0.9
params["c_rd_ed_area"] = "#999999"
params["a_rd_ed_area"] = 0.75

params["c_ed"] = "#0080c0"  # Colour ba for emp demand in rd-ed comp
params["c_rd"] = "#f58231"  # Colour ba for rand demand in rd-ed comp
params["lw_ed"] = 0.9
params["lw_rd"] = 0.9
params["c_rd_ed_area"] = "#999999"
params["a_rd_ed_area"] = 0.75

params["cmap_city_comp"] = "tab20"
