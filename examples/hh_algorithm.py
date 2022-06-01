from params import params
from paths import paths
from BikePathNet.main.algorithm import run_simulation
from multiprocessing import Pool, cpu_count


def run_city(city_params):
    place, save = city_params
    run_simulation(place, save, params=params, paths=paths)


def run_city_static(city_params):
    place, save = city_params

    params["dynamic routes"] = False
    run_simulation(place, f"{save}_static", params=params, paths=paths)

    params["penalty weighting"] = False
    run_simulation(place, f"{save}_static_npw", params=params, paths=paths)


if __name__ == "__main__":
    city = "Hamburg"
    base_save = "hh"
    hom_sets = 10

    # If you only want to run the algorithm for the empirical data, use this
    # command and remove the lines below.
    # run_simulation(city, base_save, params=params, paths=paths)

    city_p = [(city, f"{base_save}_hom_{i + 1}") for i in range(hom_sets)]
    city_p.insert(0, (city, base_save))

    p = Pool(processes=min(cpu_count() - 1, len(city_p)))
    data = p.map(run_city, city_p)
    p.close()

    # Small comparison with static approach
    # run_city_static((city, base_save))

    print("Run complete!")
