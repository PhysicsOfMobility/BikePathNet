from os.path import dirname, abspath, basename, join

project_dir = abspath(__file__)
while basename(project_dir) != "BikePathNet":
    project_dir = dirname(project_dir)

paths = {}
paths["project_dir"] = project_dir
paths["data_dir"] = join(project_dir, "data")
paths["input_folder"] = join(paths["data_dir"], "input")
paths["output_folder"] = join(paths["data_dir"], "output")
paths["log_folder"] = join(paths["data_dir"], "logs")
paths["polygon_folder"] = join(paths["data_dir"], "polygons")

paths["use_base_polygon"] = True
paths["save_devider"] = "_"

paths["plot_folder"] = join(project_dir, "plots")
paths["comp_folder"] = join(paths["data_dir"], "plot_data")
