"""
Parameters for simulation and analysis/plotting.
"""

params = {}
params["penalties"] = {
    "primary": 7,
    "secondary": 2.4,
    "tertiary": 1.4,
    "residential": 1.1,
}
params["street_cost"] = {"primary": 1, "secondary": 1, "tertiary": 1, "residential": 1}

params["penalty weighting"] = True  # Penalties for load weighting
params["dynamic routes"] = True  # Route recalculation after each step
params["forward"] = False  # Starting from scratch and adding bike paths
params["ps routes"] = False

params["cut"] = True  # If results should be normalised to first removal
params["bike_paths"] = None  # Additional ike Paths for current state
params["correct_area"] = True  # Correction of the area size
params["plot_evo"] = False  # If BP evolution should be plotted
params["evo_for"] = []  # For given modes

params["dpi"] = 300  # dpi of plots
params["titles"] = True  # If figure title should be plotted
params["legends"] = True  # If legends should be plotted
params["plot_format"] = "png"  # Format of the saved plot

params["fs_title"] = 12  # Fontsize for title
params["fs_axl"] = 7  # Fontsize for axis labels
params["fs_ticks"] = 6  # Fontsize for axis tick numbers
params["fs_legend"] = 6  # Fontsize for legends

params["figs_ba_cost"] = (2.8, 2.6)  # Figsize ba-cost plot
params["c_ba"] = "#0080c0"  # Colour for ba in normal plot
params["lw_ba"] = 0.75
params["m_ba"] = "o"  # Marker for ba
params["ms_ba"] = 2  # Marker size for ba
params["c_cost"] = "#4d4d4d"  # Colour for cost in normal plot
params["lw_cost"] = 0.75
params["m_cost"] = "s"
params["ms_cost"] = 2  # Marker size for cost

params["c_st"] = {
    "primary": "#000000",
    "secondary": "#1a1a1a",
    "tertiary": "#666666",
    "residential": "#cccccc",
    "bike paths": "#0080c0",
}
params["figs_comp_st"] = (0.8, 1.6)  # Figsize for comp_st_driven plot

params["figs_snetwork"] = (7, 5)
params["ns_snetwork"] = 4
params["nc_snetwork"] = "#333333"
params["ew_snetwork"] = 1
params["ec_snetwork"] = "#b3b3b3"

params["figs_station_usage"] = (2.7, 2.6)
params["figs_station_usage_hist"] = (2.9, 1)  # (2.4, 0.8)
params["stat_usage_norm"] = 1
params["cmap_nodes"] = "cool"  # Cmap for nodes usage
params["ns_station_usage"] = 10
params["edge_lw"] = 0.5
params["ec_station_usage"] = "#b3b3b3"

params["figs_bp_evo"] = (2.2, 2.2)
params["lw_legend_bp_evo"] = 4

params["figs_bp_comp"] = (2.8, 2.6)
params["ns_bp_comp"] = 5
params["nc_pb_evo"] = "#d726ff"
params["color_algo"] = "#0080c0"
params["color_cs"] = "#000000"
params["color_both"] = "#ff7d00"
params["color_unused"] = "#b3b3b3"


params["figs_hom_comp"] = (2.6, 2.4)
params["c_ed"] = "#0080c0"
params["c_hom"] = "#f58231"
params["lw_ed"] = 0.9
params["lw_hom"] = 0.9
params["c_hom_ed_area"] = "#999999"
params["a_hom_ed_area"] = 0.75
