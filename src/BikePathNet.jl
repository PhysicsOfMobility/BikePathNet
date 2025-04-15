"""
This module implements the code to run the bikeability algorithms.

It exports the following names:

$(EXPORTS)

It depends on the following modules:

$(IMPORTS)
"""
module BikePathNet

using DrWatson
using DocStringExtensions
using EzXML
using JSON
using Setfield
using DataStructures
using Dates
using SparseArrays
using Serde
using Random
using LoggingExtras
using Base.Threads
using ProgressBars

include("docstrings.jl")

function __init__()
    # TODO: maybe set up pycall here?
end


# setup abstract types as early as possible
"""
Supertype for all minmode specifications.
To implement your own minmode `MyMinmode`, you need to define it as a subtype
of this type, as well as the
[`init_minmode_state(alg_conf::AlgorithmConfig{MyMinmode}, shortest_path_states, g)`](@ref BikePathNet.init_minmode_state)
interface method. (See also the [Minmodes documentation](extending_bike_path_net/1_minmodes.md).)
"""
abstract type AbstractMinmode end

"""
Supertype for all minmode states.
To implement your own minmode state `MyMinmodeState` for your minmode `MyMinmode`,
you need to define it as a subtype of this type, together with the interface methods:
- [`path_loads(state::MyMinmodeState, shortest_path_state)`](@ref BikePathNet.path_loads)
- [`set_path_loads!(state::MyMinmodeState, path_id, path_loads)`](@ref BikePathNet.set_path_loads!)
- [`_edge_load(e, minstate::MyMinmodeState)`](@ref BikePathNet._edge_load)
- [`_edge_load(e, seg, minstate::MyMinmodeState)`](@ref BikePathNet._edge_load)
(See also the [Minmodes documentation](extending_bike_path_net/1_minmodes.md).)
"""
abstract type AbstractMinmodeState end
abstract type AbstractGraph end

"""
Supertype for all shortest path aggregators.
To implement your own shortest path aggregator `MyAggregator`, you need to define it as a subtype
of this type, together with the interface methods:
- `MyAggregator(g::AbstractGraph, source)`
- [`aggregate_dijkstra_step!(agg::MyAggregator, state::ShortestPathState, current_edge, current_node)`](@ref BikePathNet.aggregate_dijkstra_step!)
- [`reset_aggregator!(agg::MyAggregator, source)`](@ref BikePathNet.reset_aggregator!)
(See also the [Shortest Path Aggregators documentation](extending_bike_path_net/2_shortest_path_aggregators.md).)
"""
abstract type AbstractAggregator end

include("logging.jl")

include("graph.jl")
export Graph, SegmentGraph, SpatialGraph
include("graph_measures.jl")
export bike_path_percentage, total_cost

include("cyclist.jl")
export Cyclist

include("trip.jl")
export Trip, VariableTrip, number_of_users, load_trips

include("algorithm_config.jl")
export AlgorithmConfig

include("experiment.jl")
export Experiment, save_experiment, load_experiment
export graph_file, trips_file, output_file, log_file

include("graph_loading.jl")
export load_graph

include("graph_preparation.jl")
export prepare_graph!

include("penalties.jl")

include("minmodes.jl")
export MinPureCyclists, MinPenaltyWeightedCyclists, MinTotalDetour
export MinPenaltyCostWeightedCyclists, MinPenaltyLengthWeightedCyclists
export MinPenaltyLengthCostWeightedCyclists
export TravelTimeBenefits, TravelTimeBenefitsCost
export TravelTimeBenefitsHealthBenefits, TravelTimeBenefitsHealthBenefitsCost

include("params_io.jl")

include("shortest_paths.jl")
export OnStreetAggregator, RealDistanceAggregator, EmptyDataAggregator
export CombinedAggregator, EverythingAggregator
export ShortestPathState, build_dynamic_variables, solve_shortest_paths!
export shortest_paths

include("induced_demand.jl")
include("loads_on_streets.jl")
include("algorithm.jl")
export core_algorithm, save_core_algorithm_result, load_core_algorithm_result, run_simulation

include("comparison_state.jl")
export calculate_optimal_comparison_state, calculate_base_comparison_state
export calculate_comparison_state
export save_comparison_state_result, load_comparison_state_result

end  # BikePathNet