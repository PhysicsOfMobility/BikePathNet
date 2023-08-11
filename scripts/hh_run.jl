using DrWatson
@quickactivate "BikePathNet"

include(srcdir("node.jl"))
include(srcdir("edge.jl"))
include(srcdir("graph.jl"))
include(srcdir("shortestpath.jl"))
include(srcdir("trip.jl"))
include(srcdir("algorithm.jl"))
include(srcdir("algorithm_helper.jl"))
include(srcdir("logging_helper.jl"))

include("setup_params.jl")
include("setup_paths.jl")

city = "Hamburg"
save = "hh"

# Number of different cylcists types. You have to adapt the vecotrs of speed and penalties to fit the number of cyclist types.
# Should match the number of different cyclist types in the prep script.
params["cyclist_types"] = 1

# Speed of the cylcists (in m/s).
params["speeds"] = [1]

# Reduce the speed by the penalty factor.
params["car_penalty"] = Dict([("primary", [1/7]), ("secondary", [1/2.4]), ("tertiary", [1/1.4]), ("residential", [1/1.1])])
params["slope_penalty"] = Dict([(0.06, [1/3.2]), (0.04, [1/2.2]), (0.02, [1/1.4]), (0.0, [1/1.0])])

# Add a constant time penalty to the traveltime along an edge.
params["intersection_penalty"] = Dict([("large", [25]), ("medium", [15]), ("small", [5])])
params["turn_penalty"] = Dict([("large", [20]), ("medium", [10]), ("small", [0])])
params["bp_end_penalty"] = Dict([
    ("primary", Dict([(1, [10]), (2, [15]), (3, [25])])),
    ("secondary", Dict([(1, [5]), (2, [10]), (3, [15])])),
    ("tertiary", Dict([(1, [0]), (2, [5]), (3, [7.5])])),
    ("residential", Dict([(1, [0]), (2, [1]), (3, [1])])),
    ])

# Algorithm modes (build from scratch, how to choose minmally used edge, incorporate existing infrastructure).
modes = [(false, 1, false), (false, 1, true)]

for mode in modes
    create_algorithm_params_file(save, params, paths, mode)
end

for mode in modes
    mode_str = "$(Int(mode[1]))$(mode[2])$(Int(mode[3]))"
    run_simulation(city, save, datadir("input", save, ""), datadir("output", save, ""), datadir("input", save, "$(save)_algorithm_params_$(mode_str).json"),  datadir("logs", ""))
end