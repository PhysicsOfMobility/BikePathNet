import warnings
import sys

sys.path.append("..")

from os.path import join
from src.data import prep_city
from setup_params import params
from setup_paths import paths

warnings.filterwarnings("ignore")

city_name = "Hamburg"
save = "hh"

input_csv = join(paths["project_dir"], "data", "csvs", "hh_cleaned.csv")

# How the different cyclists types are split. Standard: one type, therefore 100% of the trips fall on that type.
# Multiple cyclist types e.g. {1: 0.25, 2: 0.5, 3: 0.25}
params["cyclist_split"] = {1: 1.0}

prep_city(
    city_name,
    save,
    input_csv,
    trunk=False,
    consolidate=True,
    tol=35,
    by_bbox=False,
    by_city=False,
    by_polygon=True,
    paths=paths,
    params=params,
)
