using DrWatson
@quickactivate "BikePathNet"

paths = Dict()
paths["input_folder"] = datadir("input")
paths["output_folder"] = datadir("output")
paths["log_folder"] = datadir("logs")
paths["polygon_folder"] = datadir("polygons")

paths["use_base_polygon"] = true
paths["save_devider"] = "_"

paths["plot_folder"] = plotsdir()
paths["comp_folder"] = datadir("plot_data")