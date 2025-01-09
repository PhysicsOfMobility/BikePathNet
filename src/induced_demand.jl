# set_baseline_demand(trips::Dict{Int,Vector{Trip}}, g, alg_config) = trips
function set_baseline_demand(trips::Dict{Int,Vector{Trip}}, g, alg_config)
    g = deepcopy(g)
    baseline_config = @set alg_config.buildup = true
    prepare_graph!(g, baseline_config)
    sps = create_shortest_path_states(g, trips; aggregator=EverythingAggregator)

    trips_new = Dict{Int,Vector{Trip}}()
    for state in sps
        for trip in state.trips
            if !haskey(trips_new, trip.origin)
                trips_new[trip.origin] = []
            end
            trip_new = @set trip.t_0 = state.felt_length[trip.destination]
            push!(trips_new[trip_new.origin], trip_new)
        end
    end
    return trips_new
end

"""
Returns new trips based on `trips`, with the baseline demand set for the case of no bike paths,
using the unprepared graph `g` and [`AlgorithmConfig`](@ref) `alg_config`. (Does nothing if `eltype(trips) == Trip`)
"""
function set_baseline_demand(trips::Dict{Int,Vector{VariableTrip}}, g, alg_config)
    g = deepcopy(g)
    baseline_config = @set alg_config.buildup = true
    prepare_graph!(g, baseline_config)
    sps = create_shortest_path_states(g, trips; aggregator=EverythingAggregator)

    trips_new = Dict{Int,Vector{VariableTrip}}()
    for state in sps
        for trip in state.trips
            if !haskey(trips_new, trip.origin)
                trips_new[trip.origin] = []
            end
            trip_new = @set trip.beta = alg_config.beta
            trip_new = @set trip_new.p_bike_0 = exp(trip_new.beta * state.felt_length[trip_new.destination])
            trip_new = @set trip_new.t_0 = state.felt_length[trip_new.destination]
            trip_new = @set trip_new.d_0 = state.data_aggregator.real_distance.on_all[trip_new.destination]
            push!(trips_new[trip_new.origin], trip_new)
        end
    end
    return trips_new
end