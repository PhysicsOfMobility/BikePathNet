# MARK: TotalRealDistances
"""
Container for all total real distances travelled by all cyclists at each step
of the algorithm. The distances are resolved by street type. (See [`real_distance_traveled_on`](@ref))
"""
struct TotalRealDistances
    on_all::Vector{Float64}
    on_street::Vector{Float64}
    on_primary::Vector{Float64}
    on_secondary::Vector{Float64}
    on_tertiary::Vector{Float64}
    on_residential::Vector{Float64}
    on_bike_path::Vector{Float64}
end
"Graph constructor for [`TotalRealDistances`](@ref)."
TotalRealDistances(g) = TotalRealDistances(ne(g) + 1)
"Run length constructor for [`TotalRealDistances`](@ref)."
function TotalRealDistances(run_length::Int)
    return TotalRealDistances(
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length)
    )
end

"""
Set, in the [`result::TotalRealDistances`](@ref TotalRealDistances),
the real distances travelled by all cyclists in the `shortest_path_states`
for the current `iteration` of the algorithm.
"""
function aggregate_result!(result::TotalRealDistances, shortest_path_states, iteration)
    result.on_primary[iteration] = sum(real_distance_traveled_on(state, :primary) for state in shortest_path_states)
    result.on_secondary[iteration] = sum(real_distance_traveled_on(state, :secondary) for state in shortest_path_states)
    result.on_tertiary[iteration] = sum(real_distance_traveled_on(state, :tertiary) for state in shortest_path_states)
    result.on_residential[iteration] = sum(real_distance_traveled_on(state, :residential) for state in shortest_path_states)
    result.on_bike_path[iteration] = sum(real_distance_traveled_on(state, :bike_path) for state in shortest_path_states)

    result.on_street[iteration] = result.on_primary[iteration] +
                                  result.on_secondary[iteration] +
                                  result.on_tertiary[iteration] +
                                  result.on_residential[iteration]

    result.on_all[iteration] = result.on_street[iteration] + result.on_bike_path[iteration]
end

# MARK: Edge and Segment Gains
const GainsTripID = @NamedTuple{origin::Int, destination::Int, cyclist_id::Int}
function deser_trip_id(k)
    k_dict = parse_json(k)
    return GainsTripID((k_dict["origin"], k_dict["destination"], k_dict["cyclist_id"]))
end

"""
Container for change in total felt length when editing each edge, resolved by cyclist and trip.
If the change is zero, nothing is stored.
Used when the graph is a [`Graph`](@ref).
"""
struct EdgeGains
    gains::Dict{Tuple{Int,Int},Dict{GainsTripID,Float64}}
end
"Empty constructor for [`EdgeGains`](@ref)."
EdgeGains() = EdgeGains(Dict())

function Serde.deser(::Type{EdgeGains}, ::Type{Dict{Tuple{Int,Int},Dict{GainsTripID,Float64}}}, val_dict)
    return Dict((parse_json(k)...,) => Dict(deser_trip_id(k2) => v2 for (k2, v2) in v) for (k, v) in val_dict)
end

"""
Container for change in total felt length when editing each segment, resolved by cyclist and trip.
If the change is zero, nothing is stored.
Used when the graph is a [`SegmentGraph`](@ref).
"""
struct SegmentGains
    gains::Dict{Int,Dict{GainsTripID,Float64}}
end
"Empty constructor for [`SegmentGains`](@ref)."
SegmentGains() = SegmentGains(Dict())

function Serde.deser(::Type{SegmentGains}, ::Type{Dict{Int,Dict{GainsTripID,Float64}}}, val_dict)
    return Dict(parse_json(k) => Dict(deser_trip_id(k2) => v2 for (k2, v2) in v) for (k, v) in val_dict)
end

function Serde.SerJson.ser_value(::Type{<:Union{EdgeGains,SegmentGains}}, ::Val{:gains}, v)
    return Dict(to_json(k1) => Dict(to_json(k2) => v2 for (k2, v2) in v1) for (k1, v1) in v)
end

"""
Add the gains in felt length caused by editing the element `min_element` to the `gains` container.
"""
function aggregate_gains!(gains, min_element, state, lengths_before, lengths_after)
    gains_key = to_gains_key(min_element)
    if !haskey(gains.gains, gains_key)
        gains.gains[gains_key] = Dict()
    end
    for trip in state.trips
        before = lengths_before[trip.destination]
        after = lengths_after[trip.destination]
        gain = after - before
        gain == 0.0 && continue
        tripid = (origin=trip.origin, destination=trip.destination, cyclist_id=trip.cyclist.id)
        gains.gains[gains_key][tripid] = gain
    end
    gains

end

"Create gains container from a [`Graph`](@ref) or [`SegmentGraph`](@ref)."
init_gains(g::Graph) = EdgeGains()
init_gains(g::SegmentGraph) = SegmentGains()

# MARK: CoreAlgorithmResult
"""
Result of running the [`core_algorithm`](@ref). Tracks all relevant data for each iteration and edge.
"""
mutable struct CoreAlgorithmResult{T}
    # per step data
    "cost of the edge edited at each step"
    total_cost::Vector{Float64}
    "percentage of bike path in the network at each step"
    bike_path_percentage::Vector{Float64}
    "real distances traveled by all cyclists at each step, resolved by street type"
    total_real_distances_traveled::TotalRealDistances
    "total felt distance traveled by all cyclists at each step"
    total_felt_distance_traveled::Vector{Float64}
    "total utility for all cyclists at each step"
    total_utility::Vector{Float64}
    "number of cyclists which travel on a street without a bike path anywhere on their trip, at each step of the algorithm"
    number_on_street::Vector{Float64}

    # edge data for other things
    "Vector of all edges that have been edited, in order"
    edited_edges::Vector{Tuple{Int,Int}}
    "Vector of all segments that have been edited, in order"
    edited_segments::Vector{Int}
    "used primary and secondary edges with a bike path when the algorithm first encounters a used edge"
    used_primary_secondary_edges::Vector{Tuple{Int,Int}}
    "all edges which have not been used by cyclists"
    unused_edges::Vector{Tuple{Int,Int}}

    "number of used edges"
    cut::Int

    "gains as a result of editing the edges/segments"
    gains::T
end

"""
    CoreAlgorithmResult(g)

Graph constructor for [`CoreAlgorithmResult`](@ref).
"""
CoreAlgorithmResult(g) = CoreAlgorithmResult(ne(g) + 1, init_gains(g))

"""
    CoreAlgorithmResult(run_length::Int, gains=EdgeGains())

Run length constructor for [`CoreAlgorithmResult`](@ref).
"""
function CoreAlgorithmResult(run_length::Int, gains=EdgeGains())
    alg_res = CoreAlgorithmResult(
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        TotalRealDistances(run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        zeros(Float64, run_length),
        Tuple{Int,Int}[],
        Int[],
        Tuple{Int,Int}[],
        Tuple{Int,Int}[],
        0,
        gains
    )
end

function Base.show(io::IO, ::MIME"text/plain", result::T) where {T<:CoreAlgorithmResult}
    println(io, T, " with ", length(result.total_cost), " iterations.")
end

######## deserialisation of result ########
function Serde.deser(::Type{<:CoreAlgorithmResult}, T::Type{Vector{Tuple{Int,Int}}}, data)
    return T([(d...,) for d in data])
end


"""
Aggregate the current state of the algorithm into the [`result::CoreAlgorithmResult`](@ref CoreAlgorithmResult).
"""
function aggregate_result!(result::CoreAlgorithmResult, g, shortest_path_states, iteration)
    result.bike_path_percentage[iteration] = bike_path_percentage(g)
    aggregate_result!(result.total_real_distances_traveled, shortest_path_states, iteration)

    result.total_felt_distance_traveled[iteration] = sum(total_felt_distance_traveled(state) for state in shortest_path_states)
    result.total_utility[iteration] = sum(utility(state) for state in shortest_path_states)
    result.number_on_street[iteration] = sum(number_on_street(state) for state in shortest_path_states)
end

# MARK: core algorithm helpers
"""
Set all relevant values in [`result::CoreAlgorithmResult`](@ref CoreAlgorithmResult) when the algorithm
first encounters a used edge or segment.
"""
function handle_first_used!(result, g, min_loaded_element, algorithm_config)
    for (k, e) in edges(g)
        if algorithm_config.use_bike_highways
            !e.bike_highway && push!(result.used_primary_secondary_edges, k)
        else
            if e.bike_path && (e.street_type in [:primary, :secondary])
                push!(result.used_primary_secondary_edges, k)
            end
        end
    end

    # copy over unused edges and segments
    for e in result.edited_edges
        push!(result.unused_edges, e)
    end
    cut = ne(g) - length(result.unused_edges)
    result.cut = cut
    @info "First used bike path (id: $(to_gains_key(min_loaded_element))) after $(length(result.unused_edges)) iterations."
end


"""
Get bike path percentages at which to log progress.
"""
function logging_steps(buildup)
    return if buildup
        [i for i in range(0.1, 1.1, 11)]
    else
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01, 0, -1]
    end
end

"""
Check if the algorithm should log progress at the current bike path percentage and log if needed.
"""
function log_progress(bike_path_percentage, next_log, buildup, start_time)
    if buildup && bike_path_percentage > next_log
        timedelta = canonicalize(round(now() - start_time, Dates.Second(1)))
        @info "Reached $next_log BPP (buildup) after $(timedelta)"
        return true
    elseif !buildup && bike_path_percentage < next_log
        timedelta = canonicalize(round(now() - start_time, Dates.Second(1)))
        @info "Reached $next_log BPP (removing) after $(timedelta)"
        return true
    else
        return false
    end
end


"""
Returns, given the `path_edge_load`s caused by a single shortest path state,
if that state needs to be recalculated when editing all edges in `current_edge_iter`.
"""
function can_state_change(path_edge_load, current_edge_iter, algorithm_config)
    if algorithm_config.buildup
        # if the edge gets better, it might be used, so we have to redo everything
        return true
    else
        # if more than no people use the edge, we have to redo the routing
        return mapreduce(e -> path_edge_load[e.source, e.destination] > 0, |, current_edge_iter)
        # TODO: what about undirected?
    end
end


# MARK: core_algorithm
"""
The main algorithm, taking
- an unprepared [`Graph`](@ref) or [`SegmentGraph`](@ref) as returned by [`load_graph`](@ref),
- a dictionary of [`Trip`](@ref) or [`VariableTrip`](@ref) as returned by [`load_trips`](@ref),
- and an [`AlgorithmConfig`](@ref)
as input.

It then prepares the graph and runs the main algorithm as specified by the `algorithm_config`,
editing the graph in the process.

It returns a [`CoreAlgorithmResult`](@ref).
"""
function core_algorithm(g, trips, algorithm_config; start_time=now())
    # has to happen before graph preparation, since we need to deepcopy the original graph.
    trips = set_baseline_demand(trips, g, algorithm_config)

    prepare_graph!(g, algorithm_config)

    # create empty result
    run_length = count(!e.blocked for (od, e) in edges(g)) + 1
    result = CoreAlgorithmResult(run_length, init_gains(g))

    # setup all shortest path states, one for each cyclist and origin in trips
    @info "Initial calculation started."
    shortest_path_states = create_shortest_path_states(g, trips)

    # setup load tracking
    pure_config = @set algorithm_config.minmode = MinPureCyclists()
    path_edge_loads = init_minmode_state(pure_config, shortest_path_states, g)
    weighted_path_edge_loads = init_minmode_state(algorithm_config, shortest_path_states, g)

    # setup gains tracking
    lengths_before = [tripwise_total_felt_distance_traveled(state) for state in shortest_path_states]

    # set first entry in result
    result.total_cost[1] = 0.0
    aggregate_result!(result, g, shortest_path_states, 1)
    @info "Initial calculation ended."

    log_at = logging_steps(algorithm_config.buildup)
    next_log = popfirst!(log_at)

    is_first_used = true
    current_iteration = 2
    while current_iteration <= run_length
        min_loaded_element, is_used = minimal_loaded_element(g, weighted_path_edge_loads, algorithm_config)
        is_non_existent(min_loaded_element) && break

        if is_used && is_first_used
            handle_first_used!(result, g, min_loaded_element, algorithm_config)
            is_first_used = false
        end

        current_edges = to_edge_iter(min_loaded_element, g)

        # flip and edit all relevant edges
        recalc_needed = false
        for current_edge in current_edges
            push!(result.edited_edges, (current_edge.source, current_edge.destination))
            push!(result.edited_segments, current_edge.seg_id)

            if !current_edge.ex_inf
                result.total_cost[current_iteration] = current_edge.cost
                edit_edge!(current_edge)
                if algorithm_config.undirected
                    edit_edge!(edges(g)[current_edge.destination, current_edge.origin])
                end
                recalc_needed = true  # to rerun the shortest paths only if at least one edge got realistically edited.
            else
                current_edge.edited = true
            end
            current_iteration += 1
        end

        # recalculate all state that might have changed
        if recalc_needed
            edge_load_channel = Channel(length(shortest_path_states))
            weighted_edge_load_channel = Channel(length(shortest_path_states))
            gains_channel = Channel(length(shortest_path_states))

            solver_spawn_task = Threads.@spawn begin
                Threads.@sync for i in eachindex(shortest_path_states)
                    if can_state_change(path_edge_loads.individual_loads[i], current_edges, algorithm_config)
                        state = shortest_path_states[i]
                        Threads.@spawn begin
                            reset_shortest_path_state!(state)
                            dynamic_vars = build_dynamic_variables(state)
                            solve_shortest_paths!(state, dynamic_vars..., g)

                            current_state_loads = path_loads(path_edge_loads, state)
                            weighted_current_state_loads = path_loads(weighted_path_edge_loads, state)

                            shortest_path_state_lengths = tripwise_total_felt_distance_traveled(state)

                            put!(edge_load_channel, (i, current_state_loads))
                            put!(weighted_edge_load_channel, (i, weighted_current_state_loads))
                            put!(gains_channel, (i, shortest_path_state_lengths))
                        end
                    end
                end
                close(edge_load_channel)
                close(weighted_edge_load_channel)
                close(gains_channel)
            end

            sync_load_task = Threads.@spawn for (i, loads) in edge_load_channel
                set_path_loads!(path_edge_loads, i, loads)
            end

            sync_weighted_load_task = Threads.@spawn for (i, weighted_loads) in weighted_edge_load_channel
                set_path_loads!(weighted_path_edge_loads, i, weighted_loads)
            end

            sync_gains_task = Threads.@spawn for (i, state_lengths) in gains_channel
                aggregate_gains!(result.gains, min_loaded_element, shortest_path_states[i], lengths_before[i], state_lengths)
                lengths_before[i] = state_lengths
            end

            wait(solver_spawn_task)
            wait(sync_load_task)
            wait(sync_weighted_load_task)
            wait(sync_gains_task)
        end

        aggregate_result!(result, g, shortest_path_states, current_iteration - 1)

        # log progress if needed
        if log_progress(result.bike_path_percentage[current_iteration-1], next_log, algorithm_config.buildup, start_time)
            next_log = popfirst!(log_at)
        end
    end

    return result
end

# MARK: result IO
"""
Save the [`result::CoreAlgorithmResult`](@ref CoreAlgorithmResult) from running
[`core_algorithm`](@ref) to a `json` file stored at `file`.
"""
function save_core_algorithm_result(file, result)
    @info "Saving algorithm results..."
    mkpath(dirname(file))
    open(file, "w") do io
        write(io, to_json(result))
    end
end

"""
Load a [`CoreAlgorithmResult`](@ref) from a `json` file stored at `file`.
"""
function load_core_algorithm_result(file)
    deser_json(CoreAlgorithmResult, read(file))
end

# MARK: simulation
"""
Given the path to a `json` file containing an [`Experiment`](@ref),
sets up logging, loads the graph and trips, runs the algorithm and saves the result
as specified by the `Experiment`.
"""
function run_simulation(experiment_file)

    experiment_config = load_experiment(experiment_file)
    algorithm_config = experiment_config.algorithm_config

    if !isnothing(log_file(experiment_config))
        mkpath(dirname(log_file(experiment_config)))
        setup_logger(log_file(experiment_config), experiment_config.save)
    else
        setup_logger(experiment_config.save)
    end

    mode = "$(algorithm_config.buildup ? "buildup" : "teardown"), $(split(string(typeof(algorithm_config.minmode)), '.')[end]) and $(algorithm_config.use_existing_infrastructure ? "with existing infrastructure" : "without existing infrastructure")"

    g = load_graph(graph_file(experiment_config), experiment_config.graph_type)
    @info "Loaded graph ($(typeof(g))) with $(nv(g)) nodes and $(ne(g)) edges."
    trips = load_trips(trips_file(experiment_config), experiment_config.cyclists, experiment_config.trip_type)
    @info "Loaded $(sum([length(trip_vector) for trip_vector in values(trips)])) unique trips ($(typeof(first(values(trips))[1]))) between $(length(unique([station for trip_vector in values(trips) for trip in trip_vector for station in [trip.origin, trip.destination]]))) stations."

    start_time = now()
    @info "Starting with $mode. Using $(nthreads()) thread(s)."
    result = core_algorithm(g, trips, algorithm_config, start_time=start_time)
    @info "Finished [$mode] after $(canonicalize(round(now()-start_time, Dates.Second(1))))."

    save_core_algorithm_result(output_file(experiment_config), result)

    return result
end
