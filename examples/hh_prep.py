from BikePathNet.main.data import prep_city
from params import params
from paths import paths

city_name = "Hamburg"
save_name = "hh"
hom_sets = 10

input_csv = f"{save_name}.csv"

prep_city(
    city_name,
    save_name,
    input_csv,
    consolidate=False,
    cached_graph=True,
    cached_graph_folder=f'{paths["input_folder"]}{save_name}/',
    cached_graph_name=save_name,
    paths=paths,
    params=params,
)

# If you want to use the newest version of the street network us the following command and remove
# the one above. Be aware, the existing graph will be overwritten!
"""prep_city(
    city_name,
    save_name,
    input_csv,
    consolidate=True,
    tol=35,
    paths=paths,
    params=params,
)"""

for i in range(hom_sets):
    hom_save_name = f"{save_name}_hom_{i+1}"
    hom_input_csv = f"{hom_save_name}.csv"
    prep_city(
        city_name,
        hom_save_name,
        hom_input_csv,
        consolidate=False,
        cached_graph=True,  # use same graph as emp. demand
        cached_graph_folder=f'{paths["input_folder"]}{save_name}/',
        cached_graph_name=save_name,
        paths=paths,
        params=params,
    )

# Small comparison with static approach
"""static = f"{save_name}_static"
prep_city(
    city_name,
    static,
    input_csv,
    consolidate=False,
    cached_graph=True,
    cached_graph_folder=f'{paths["input_folder"]}{save_name}/',
    cached_graph_name=save_name,
    paths=paths,
    params=params,
)
static_npw = f"{save_name}_static_npw"
prep_city(
    city_name,
    static_npw,
    input_csv,
    consolidate=False,
    cached_graph=True,
    cached_graph_folder=f'{paths["input_folder"]}{save_name}/',
    cached_graph_name=save_name,
    paths=paths,
    params=params,
)
static_ps = f"{save_name}_static_ps"
prep_city(
    city_name,
    static_ps,
    input_csv,
    consolidate=False,
    cached_graph=True,
    cached_graph_folder=f'{paths["input_folder"]}{save_name}/',
    cached_graph_name=save_name,
    paths=paths,
    params=params,
)"""

"""# Small comparison with forward approach
forward = f"{save_name}_forward"
prep_city(
    city_name,
    forward,
    input_csv,
    consolidate=False,
    cached_graph=True,
    cached_graph_folder=f'{paths["input_folder"]}{save_name}/',
    cached_graph_name=save_name,
    paths=paths,
    params=params,
)"""
