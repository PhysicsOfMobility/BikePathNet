# MARK: TotalRealDistance
"""
Container for the total real distance travelled by all cyclists at a comparison state of the graph.
The distances are resolved by street type. (See [`real_distance_traveled_on`](@ref))
"""
struct TotalRealDistance
    on_all::Float64
    on_street::Float64
    on_primary::Float64
    on_secondary::Float64
    on_tertiary::Float64
    on_residential::Float64
    on_bike_path::Float64
end

"""
Create a new [`TotalRealDistance`](@ref) struct from a [`TotalRealDistances`](@ref)
struct at a specific `iteration`.
"""
function TotalRealDistance(real_distances::TotalRealDistances, iteration)
    return TotalRealDistance(
        real_distances.on_all[iteration],
        real_distances.on_street[iteration],
        real_distances.on_primary[iteration],
        real_distances.on_secondary[iteration],
        real_distances.on_tertiary[iteration],
        real_distances.on_residential[iteration],
        real_distances.on_bike_path[iteration]
    )
end

# MARK: ComparisonStateResult
"""
Result of running the [`calculate_comparison_state`](@ref) on a specific graph.
"""
struct ComparisonStateResult
    "total cost of building all bike paths in the graph"
    total_cost::Float64
    "percentage of bike paths in the graph (by length)"
    bike_path_percentage::Float64
    "total real distance travelled by all cyclists, resolved by street type"
    total_real_distances_traveled::TotalRealDistance
    "total felt distance travelled by all cyclists"
    total_felt_distance_traveled::Float64
    "total utility for all cyclists"
    total_utility::Float64
    "number of cyclists which travelled on a street without a bike path anywhere on their trip"
    number_on_street::Float64

    "edges which had a bike path in the comparison state"
    bike_paths::Vector{Tuple{Int,Int}}
    "loads on the edges in the comparison state"
    edge_loads::Dict{Tuple{Int,Int},Float64}
end


######## serialise edge loads #######
function Serde.SerJson.ser_value(::Type{ComparisonStateResult}, ::Val{:edge_loads}, v)
    return Dict(to_json(k) => v for (k, v) in v)
end

###### deserialise edge loads #######
function Serde.deser(::Type{ComparisonStateResult}, ::Type{Dict{Tuple{Int,Int},Float64}}, val_dict)
    return Dict((parse_json(k)...,) => v for (k, v) in val_dict)
end


"""
Given a graph `g`, `trips`, an `algo_config`, calculates the optimal comparison state for this graph,
returning a [`ComparisonStateResult`](@ref).
The optimal comparison state is characterised by the fact that all streets have bike paths
and all penalties are set to the ideal values.
"""
function calculate_optimal_comparison_state(g, trips, algo_config)
    optimal_config = AlgorithmConfig(
        buildup=false,
        use_existing_infrastructure=false,
        use_blocked_streets=false,
        add_blocked_streets=[(-1,-1)],
        use_bike_highways=false,    # TODO: add config/switch for bike highways
        save_edge_loads=false,
        undirected=false,
        minmode=MinPureCyclists(),
        beta=algo_config.beta,
    )

    cyclists = unique([trip.cyclist for trip_vector in values(trips) for trip in trip_vector])
    optimal_cyclists = Dict([
        (c.id, Cyclist(
            c.id,
            c.speed,
            Dict([(k, 1.0) for k in keys(c.car_penalty)]),
            Dict([(k, 1.0) for k in keys(c.slope_penalty)]),
            Dict([(k, 0.0) for k in keys(c.intersection_penalty)]),
            Dict([(k, 0.0) for k in keys(c.edge_time_penalty)]),
            Dict([(k, 0.0) for k in keys(c.turn_penalty)]),
            Dict([(k, Dict([(1, 0.0)])) for k in keys(c.bikepath_end_penalty)]),
            c.value_hb,
        )
        )
        for c in cyclists])

    new_trips = Dict([(trip_origin, [@set trip.cyclist = optimal_cyclists[trip.cyclist.id] for trip in trip_vector]) for (trip_origin, trip_vector) in trips])

    start_time = now()
    @info "Calculating optimal state."
    results = calculate_comparison_state(g, new_trips, optimal_config, collect(keys(edges(g))))
    @info "Finished optimal state after $(canonicalize(round(now()-start_time, Dates.Second(1))))."

    return results
end

"""
Calculates the optimal comparison state for the experiment and bike paths specified in `experiment_file`
and writing it to `output_file(experiment)`. The file is loaded with [`load_comparison_params`](@ref), 
the bike paths are ignored.
"""
function calculate_optimal_comparison_state(experiment_file::String)
    experiment_config, _ = load_comparison_params(experiment_file)

    if !isnothing(log_file(experiment_config))
        mkpath(dirname(log_file(experiment_config)))
        setup_logger(log_file(experiment_config))
    else
        setup_logger()
    end

    g = load_graph(graph_file(experiment_config), experiment_config.graph_type)
    @info "Loaded graph ($(typeof(g))) with $(nv(g)) nodes and $(ne(g)) edges."
    trips = load_trips(trips_file(experiment_config), experiment_config.cyclists, experiment_config.trip_type)
    @info "Loaded $(sum([length(trip_vector) for trip_vector in values(trips)])) unique trips ($(typeof(first(values(trips))[1]))) between $(length(unique([station for trip_vector in values(trips) for trip in trip_vector for station in [trip.origin, trip.destination]]))) stations."

    result = calculate_optimal_comparison_state(g, trips, experiment_config.algorithm_config)

    save_comparison_state_result(output_file(experiment_config), result)

    return result
end

"""
Given a graph `g`, `trips`, an `algo_config`, calculates the base comparison state for this graph,
returning a [`ComparisonStateResult`](@ref).
The base comparison state is characterised by the fact that no streets have bike paths.
"""
function calculate_base_comparison_state(g, trips, algo_config)
    base_config = AlgorithmConfig(
        buildup=false,
        use_existing_infrastructure=false,
        use_blocked_streets=false,
        add_blocked_streets=[(-1,-1)],
        use_bike_highways=false,    # TODO: add config/switch for bike highways
        save_edge_loads=false,
        undirected=false,
        minmode=MinPureCyclists(),
        beta=algo_config.beta,
    )

    start_time = now()
    @info "Calculating base state."
    results = calculate_comparison_state(g, trips, base_config, [])
    @info "Finished base state after $(canonicalize(round(now()-start_time, Dates.Second(1))))."

    return results
end

"""
Calculates the base comparison state for the experiment and bike paths specified in `experiment_file`
and writing it to `output_file(experiment)`. The file is loaded with [`load_comparison_params`](@ref), 
the bike paths are ignored.
"""
function calculate_base_comparison_state(experiment_file::String)
    experiment_config, _ = load_comparison_params(experiment_file)

    if !isnothing(log_file(experiment_config))
        mkpath(dirname(log_file(experiment_config)))
        setup_logger(log_file(experiment_config))
    else
        setup_logger()
    end

    g = load_graph(graph_file(experiment_config), experiment_config.graph_type)
    @info "Loaded graph ($(typeof(g))) with $(nv(g)) nodes and $(ne(g)) edges."
    trips = load_trips(trips_file(experiment_config), experiment_config.cyclists, experiment_config.trip_type)
    @info "Loaded $(sum([length(trip_vector) for trip_vector in values(trips)])) unique trips ($(typeof(first(values(trips))[1]))) between $(length(unique([station for trip_vector in values(trips) for trip in trip_vector for station in [trip.origin, trip.destination]]))) stations."

    result = calculate_base_comparison_state(g, trips, experiment_config.algorithm_config)

    save_comparison_state_result(output_file(experiment_config), result)

    return result
end


"""
Calculates the comparison state for the experiment and bike paths specified in `experiment_file`
and writing it to `output_file(experiment)`. The file is loaded with [`load_comparison_params`](@ref).
"""
function calculate_comparison_state(experiment_file::String)
    experiment_config, bike_paths = load_comparison_params(experiment_file)

    if !isnothing(log_file(experiment_config))
        mkpath(dirname(log_file(experiment_config)))
        setup_logger(log_file(experiment_config))
    else
        setup_logger()
    end

    g = load_graph(graph_file(experiment_config), experiment_config.graph_type)
    @info "Loaded graph ($(typeof(g))) with $(nv(g)) nodes and $(ne(g)) edges."
    trips = load_trips(trips_file(experiment_config), experiment_config.cyclists, experiment_config.trip_type)
    @info "Loaded $(sum([length(trip_vector) for trip_vector in values(trips)])) unique trips ($(typeof(first(values(trips))[1]))) between $(length(unique([station for trip_vector in values(trips) for trip in trip_vector for station in [trip.origin, trip.destination]]))) stations."

    start_time = now()
    @info "Calculating comparison state."
    result = calculate_comparison_state(g, trips, experiment_config.algorithm_config, bike_paths)
    @info "Finished comparison state after $(canonicalize(round(now()-start_time, Dates.Second(1))))."

    save_comparison_state_result(output_file(experiment_config), result)

    return result
end

"""
Given a graph `g`, `trips`, an `algorithm_config` and a list of `bike_paths`,
calculates the comparison state for this graph, returning a [`ComparisonStateResult`](@ref).
This comparison state is used as a comparison point for the [`core_algorithm`](@ref).
"""
function calculate_comparison_state(g, trips, algorithm_config, bike_paths)
    # we do not care about save_edge_loads, undirected, minmode
    # in the algorithm_config struct. (should we build a comparison_config struct?)

    # TODO: just that no one expects different things when changing these fields of the config.
    @assert algorithm_config.buildup == false "buildup has to be false for the comparison state"
    @assert algorithm_config.save_edge_loads == false "save_edge_loads should be false for the comparison state"
    @assert algorithm_config.undirected == false "undirected should be false for the comparison state"
    @assert algorithm_config.minmode == MinPureCyclists() "minmode should be MinPureCyclists for the comparison state"

    trips = set_baseline_demand(trips, g, algorithm_config)

    prepare_graph!(g, algorithm_config)
    for (k, e) in edges(g)
        if algorithm_config.use_existing_infrastructure && e.ex_inf && !(k in bike_paths)
            push!(bike_paths, k)
        end
        if !(k in bike_paths) && !e.blocked  # blocked state gets set in prepare_graph! depending on the config
            edit_edge!(e)
        end
    end

    shortest_path_states = create_shortest_path_states(g, trips)
    path_edge_loads = init_minmode_state(algorithm_config, shortest_path_states, g)

    # reuse core result and aggregation thereof
    core_result = CoreAlgorithmResult(1)

    core_result.total_cost[1] = total_cost(g, bike_paths, algorithm_config.use_existing_infrastructure)

    aggregate_result!(core_result, g, shortest_path_states, 1)

    edge_loads = Dict(k => path_edge_loads.aggregated_loads[k...] for k in keys(edges(g)))

    # copy over stuff to new comparison state result
    real_distance = TotalRealDistance(core_result.total_real_distances_traveled, 1)

    return ComparisonStateResult(
        core_result.total_cost[1],
        core_result.bike_path_percentage[1],
        real_distance,
        core_result.total_felt_distance_traveled[1],
        core_result.total_utility[1],
        core_result.number_on_street[1],
        bike_paths,
        edge_loads
    )
end


"""
Save the [`result::ComparisonStateResult`](@ref ComparisonStateResult)
to a `json` file stored at `file`.
"""
function save_comparison_state_result(file, result)
    @info "Saving comparison results..."
    mkpath(dirname(file))
    open(file, "w") do io
        write(io, to_json(result))
    end
end

"""
Load a [`ComparisonStateResult`](@ref) from a `json` file stored at `file`.
"""
function load_comparison_state_result(file)
    deser_json(ComparisonStateResult, read(file))
end

"""
Get the `used_primary_secondary_edges` from the a [`CoreAlgorithmResult`](@ref)
stored in `file`.
"""
function get_ps_bp_from_algorithm_results(file)
    data = deser_json(CoreAlgorithmResult, read(file))
    return data.used_primary_secondary_edges
end

"""
Get the edges constructed so far from the a [`CoreAlgorithmResult`](@ref)
stored in `algo_file` for the same bike path percentage as in the `comp_file`.
"""
function get_comparison_bp_from_algorithm_results(algo_file, comp_file, buildup=false)
    algo_data = deser_json(CoreAlgorithmResult, read(algo_file))
    comp_data = deser_json(ComparisonStateResult, read(comp_file))

    if buildup
        bike_paths = algo_data.edited_edges
        bpp_algo = algo_data.bike_path_percentage
    else
        bike_paths = reverse(algo_data.edited_edges)
        bpp_algo = reverse(algo_data.bike_path_percentage)
    end

    idx = partialsortperm(abs.(bpp_algo .- comp_data.bike_path_percentage), 1)

    return bike_paths[1:idx]
end