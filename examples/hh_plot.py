import warnings
from params import params
from paths import paths
from BikePathNet.main.plot import plot_city, plot_city_hom_demand,\
    plot_dynamic_vs_static

warnings.simplefilter(action='ignore', category=FutureWarning)

base_save = 'hh'
city = 'Hamburg'

params["legends"] = False
params["titles"] = False
params["plot_format"] = 'png'

params["figs_station_usage"] = (2.7, 2.6)
params["stat_usage_norm"] = 1246/365    # norm station usage to year average

params["figs_bp_comp"] = (2.8, 2.6)

plot_city(city, base_save, paths=paths, params=params)

for i in range(1, 11):
    hom_save = f'{base_save}_hom_{i}'
    params["stat_usage_norm"] = 1
    plot_city(city, hom_save, paths=paths, params=params)

plot_city_hom_demand(city, base_save, paths, params, hom_pattern='hom',
                     nbr_of_hom_sets=10)


# Small comparison with static approach
# plot_city(city, f'{base_save}_static', paths=paths, params=params)
# plot_city(city, f'{base_save}_static_npw', paths=paths, params=params)
# plot_dynamic_vs_static(base_save, paths, params)
