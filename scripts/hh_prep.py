import sys
from os.path import join
from setup_paths import paths
from setup_params import params

sys.path.append(paths["project_dir"])
from src.prep_and_plot import prep_city


city_name = "Hamburg"
city_save = "hh"

input_csv = join(paths["data_dir"], "csvs", "hh_cleaned.csv")
polygon_file = join(paths["data_dir"], "polygons", "hh.json")

prep_city(
    city_name,
    city_save,
    input_csv,
    polygon_file,
    trunk=False,
    consolidate=True,
    tol=10,
    by_bbox=False,
    by_city=False,
    by_polygon=True,
    paths=paths,
    params=params,
)
