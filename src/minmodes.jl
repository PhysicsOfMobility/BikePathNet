# MARK: Minmode definitions
"Minmode using the pure number of cyclists as the load on a street."
struct MinPureCyclists <: AbstractMinmode end
"Minmode using the number of cyclist divided by the [`car_penalty`](@ref) as the load."
struct MinPenaltyWeightedCyclists <: AbstractMinmode end
"Minmode using the number of cyclist divided by the [`car_penalty`](@ref) and cost of the street/segment as the load."
struct MinPenaltyCostWeightedCyclists <: AbstractMinmode end
"Minmode using the number of cyclist divided by the [`car_penalty`](@ref), multiplied by the `edge.length` as the load."
struct MinPenaltyLengthWeightedCyclists <: AbstractMinmode end
"Minmode using the number of cyclist divided by the [`car_penalty`](@ref), and cost of the street/segment, multiplied by the `edge.length` as the load."
struct MinPenaltyLengthCostWeightedCyclists <: AbstractMinmode end
"Minmode using the consumer surplus as the load."
struct TravelTimeBenefits <: AbstractMinmode end
"Minmode using the consumer surplus, divided by the cost of the street/segment as the load."
struct TravelTimeBenefitsCost <: AbstractMinmode end
"Minmode using the consumer surplus and health benefits as the load."
struct TravelTimeBenefitsHealthBenefits <: AbstractMinmode end
"Minmode using the consumer surplus and health benefits, divided by the cost of the street/segment as the load."
struct TravelTimeBenefitsHealthBenefitsCost <: AbstractMinmode end

"Minmode using the total added/removed felt detour when removing/adding a bike path on a street as the load."
struct MinTotalDetour <: AbstractMinmode end

"Minmode using a predefined fixed order."
struct FixedOrderMinmode{T} <: AbstractMinmode
    order::Vector{T}
end

"""
    init_minmode_state(alg_conf::AlgorithmConfig{T}, shortest_path_states, g) where T<:AbstractMinmode

Given the algorithm configuration `alg_conf`, a vector of SOLVED `shortest_path_states` and a graph `g`,
returns the minmode state associated to minmode `T` with all loads set correctly.
"""
function init_minmode_state end
# forward only the graph if g is segment graph
init_minmode_state(alg_conf, shortest_path_states, g::SegmentGraph) = init_minmode_state(alg_conf, shortest_path_states, g.graph)

"""
    path_loads(state::AbstractMinmodeState, shortest_path_state)

Given the minmode state `state` and a SOLVED `shortest_path_state`, returns an object which holds
the loads on the streets caused by the trips in the `shortest_path_state`.
!!! danger "This function has to be thread-safe"
    Do not mutate any shared object that might be stored in the `state`,
    as this function can be called in parallel for different `shortest_path_state`s.
"""
function path_loads end

"""
    set_path_loads!(state::AbstractMinmodeState, path_id, path_loads)

Given the minmode state `state`, the `path_id` (a one based index of the
shortest path state used to calculate the `path_loads`) and the `path_loads` object
returned by [`path_loads`](@ref), updates the `state` with the new `path_loads`.
"""
function set_path_loads! end


# MARK: MinPureCyclistsState
struct MinPureCyclistsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
end

function init_minmode_state(::AlgorithmConfig{MinPureCyclists}, shortest_path_states, ::Graph)
    num_nodes = length(shortest_path_states[1].felt_length)
    aggregated_loads = spzeros(num_nodes, num_nodes)
    individual_loads = [spzeros(num_nodes, num_nodes) for _ in shortest_path_states]

    minmode_state = MinPureCyclistsState(aggregated_loads, individual_loads)

    for (i, shortest_path_state) in enumerate(shortest_path_states)
        loads = path_loads(minmode_state, shortest_path_state)
        set_path_loads!(minmode_state, i, loads)
    end
    return minmode_state
end

function path_loads(state::MinPureCyclistsState, shortest_path_state)
    loads = spzeros(size(state.aggregated_loads))
    for trip in shortest_path_state.trips
        dst = trip.destination
        while dst != trip.origin
            src = shortest_path_state.predecessor[dst]
            loads[src, dst] += number_of_users(trip, shortest_path_state)
            dst = src
        end
    end
    return loads
end

function set_path_loads!(state::MinPureCyclistsState, path_id, path_loads)
    state.aggregated_loads .-= state.individual_loads[path_id]
    state.aggregated_loads .+= path_loads

    state.individual_loads[path_id] = path_loads
    return state
end


# MARK: Structurally similar
# TODO: these could be refactored to something like
# struct MinStateWithGraph{AbstractMinmode} <: AbstractMinmodeState
struct MinPenaltyWeightedCyclistsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct MinPenaltyCostWeightedCyclistsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct MinPenaltyLengthWeightedCyclistsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct MinPenaltyLengthCostWeightedCyclistsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct TravelTimeBenefitsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct TravelTimeBenefitsCostState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct TravelTimeBenefitsHealthBenefitsState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end
struct TravelTimeBenefitsHealthBenefitsCostState <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    g::Graph
end

function _weighted_trip_load(_::MinPenaltyWeightedCyclistsState, trip, shortest_path_state, _, penalty_weight)
    number_of_users(trip, shortest_path_state) / penalty_weight
end
function _weighted_trip_load(_::MinPenaltyCostWeightedCyclistsState, trip, shortest_path_state, _, penalty_weight)
    number_of_users(trip, shortest_path_state) / penalty_weight
end
function _weighted_trip_load(_::MinPenaltyLengthWeightedCyclistsState, trip, shortest_path_state, e, penalty_weight)
    number_of_users(trip, shortest_path_state) / penalty_weight * e.length
end
function _weighted_trip_load(_::MinPenaltyLengthCostWeightedCyclistsState, trip, shortest_path_state, e, penalty_weight)
    number_of_users(trip, shortest_path_state) / penalty_weight * e.length
end
function _weighted_trip_load(_::TravelTimeBenefitsState, trip, shortest_path_state, e, penalty_weight)
    0.5 * trip.vot * (number_of_users(trip, shortest_path_state) + trip.number_of_users) * e.length * (1 / trip.cyclist.speed) * ((1 / penalty_weight) - 1)
end
function _weighted_trip_load(_::TravelTimeBenefitsCostState, trip, shortest_path_state, e, penalty_weight)
    0.5 * trip.vot * (number_of_users(trip, shortest_path_state) + trip.number_of_users) * e.length * (1 / trip.cyclist.speed) * ((1 / penalty_weight) - 1)
end
function _weighted_trip_load(_::TravelTimeBenefitsHealthBenefitsState, trip, shortest_path_state, e, penalty_weight)
    delta_te = e.length * (1 / trip.cyclist.speed) * ((1 / penalty_weight) - 1)
    beta = trip.beta
    P_0 = trip.p_bike_0 / (trip.p_bike_0 + trip.p_other)
    p_t = exp(trip.beta * shortest_path_state.felt_length[trip.destination])
    P_t = p_t / (p_t + trip.p_other)
    n_0 = trip.number_of_users
    n = P_t / P_0 * n_0
    lambda = shortest_path_state.data_aggregator.real_distance.on_all[trip.destination]
    delta_CS = trip.vot * 0.5 * (beta*n*(trip.t_0-shortest_path_state.felt_length[trip.destination])*(P_t-1) + (n_0+n))
    delat_HB = trip.cyclist.value_hb * beta * n * lambda * (P_t-1)
    return (delta_CS + delat_HB) * delta_te
end
function _weighted_trip_load(_::TravelTimeBenefitsHealthBenefitsCostState, trip, shortest_path_state, e, penalty_weight)
    delta_te = e.length * (1 / trip.cyclist.speed) * ((1 / penalty_weight) - 1)
    beta = trip.beta
    P_0 = trip.p_bike_0 / (trip.p_bike_0 + trip.p_other)
    p_t = exp(trip.beta * shortest_path_state.felt_length[trip.destination])
    P_t = p_t / (p_t + trip.p_other)
    n_0 = trip.number_of_users
    n = P_t / P_0 * n_0
    l = shortest_path_state.data_aggregator.real_distance.on_all[trip.destination]
    delta_CS = trip.vot * 0.5 * (beta*n*(trip.t_0-shortest_path_state.felt_length[trip.destination])*(P_t-1) + (n_0+n))
    delat_HB = trip.cyclist.value_hb * beta * n * l * (P_t-1)
    return (delta_CS + delat_HB) * delta_te
end

const MINSTATES_WITH_GRAPH = Union{
    MinPenaltyWeightedCyclistsState,
    MinPenaltyCostWeightedCyclistsState,
    MinPenaltyLengthWeightedCyclistsState,
    MinPenaltyLengthCostWeightedCyclistsState,
    TravelTimeBenefitsState,
    TravelTimeBenefitsCostState,
    TravelTimeBenefitsHealthBenefitsState,
    TravelTimeBenefitsHealthBenefitsCostState
}
const MINMODES_WITH_GRAPH = Union{
    MinPenaltyWeightedCyclists,
    MinPenaltyCostWeightedCyclists,
    MinPenaltyLengthWeightedCyclists,
    MinPenaltyLengthCostWeightedCyclists,
    TravelTimeBenefits,
    TravelTimeBenefitsCost,
    TravelTimeBenefitsHealthBenefits,
    TravelTimeBenefitsHealthBenefitsCost
}
const MINMODES_TO_MINSTATES_WITH_GRAPH = Dict(
    MinPenaltyWeightedCyclists => MinPenaltyWeightedCyclistsState,
    MinPenaltyCostWeightedCyclists => MinPenaltyCostWeightedCyclistsState,
    MinPenaltyLengthWeightedCyclists => MinPenaltyLengthWeightedCyclistsState,
    MinPenaltyLengthCostWeightedCyclists => MinPenaltyLengthCostWeightedCyclistsState,
    TravelTimeBenefits => TravelTimeBenefitsState,
    TravelTimeBenefitsCost => TravelTimeBenefitsCostState,
    TravelTimeBenefitsHealthBenefits => TravelTimeBenefitsHealthBenefitsState,
    TravelTimeBenefitsHealthBenefitsCost => TravelTimeBenefitsHealthBenefitsCostState,
)

function init_minmode_state(_::AlgorithmConfig{T}, shortest_path_states, g::Graph) where {T<:MINMODES_WITH_GRAPH}
    num_nodes = length(shortest_path_states[1].felt_length)
    aggregated_loads = spzeros(num_nodes, num_nodes)
    individual_loads = [spzeros(num_nodes, num_nodes) for _ in shortest_path_states]

    minmode_state = MINMODES_TO_MINSTATES_WITH_GRAPH[T](aggregated_loads, individual_loads, g)

    for (i, shortest_path_state) in enumerate(shortest_path_states)
        loads = path_loads(minmode_state, shortest_path_state)
        set_path_loads!(minmode_state, i, loads)
    end
    return minmode_state
end

function path_loads(state::MINSTATES_WITH_GRAPH, shortest_path_state)
    loads = spzeros(size(state.aggregated_loads))
    for trip in shortest_path_state.trips
        dst = trip.destination
        while dst != trip.origin
            src = shortest_path_state.predecessor[dst]
            e = state.g.edges[(src, dst)]
            penalty_weight = car_penalty(e, trip.cyclist)
            loads[src, dst] += _weighted_trip_load(state, trip, shortest_path_state, e, penalty_weight)
            dst = src
        end
    end
    return loads
end

function set_path_loads!(state::MINSTATES_WITH_GRAPH, path_id, path_loads)
    state.aggregated_loads .-= state.individual_loads[path_id]
    state.aggregated_loads .+= path_loads

    state.individual_loads[path_id] = path_loads
    return state
end

# MARK: MinTotalDetourState
struct MinTotalDetourState{T<:AbstractGraph} <: AbstractMinmodeState
    aggregated_detours::SparseMatrixCSC{Float64,Int64}
    individual_detours::Vector{SparseMatrixCSC{Float64,Int64}}
    algorithm_config::AlgorithmConfig
    g::T
end

function init_minmode_state(alg_conf::AlgorithmConfig{MinTotalDetour}, shortest_path_states, g::Graph)
    num_nodes = length(shortest_path_states[1].felt_length)
    aggregated_detours = spzeros(num_nodes, num_nodes)
    individual_detours = [spzeros(num_nodes, num_nodes) for _ in shortest_path_states]

    minmode_state = MinTotalDetourState(aggregated_detours, individual_detours, alg_conf, g)

    pbar = ProgressBar(collect(enumerate(shortest_path_states)), printing_delay=1.0)
    set_description(pbar, "Setting up minmode state")

    load_channel = Channel(length(shortest_path_states))

    load_task = Threads.@spawn begin
        Threads.@threads for (i, shortest_path_state) in pbar
            loads = path_loads(minmode_state, shortest_path_state)
            put!(load_channel, (i, loads))
        end
        close(load_channel)
    end
    sync_load_task = Threads.@spawn for (i, loads) in load_channel
        set_path_loads!(minmode_state, i, loads)
    end
    wait(load_task)
    wait(sync_load_task)

    return minmode_state
end

# iterator of elements that can be edited
function editable_elements(g::SegmentGraph)
    (MinLoadedSegment(k) for (k, s) in g.segments if let e = edges(g)[s.edges[1]]
        !(e.edited || e.blocked)
    end)
end
function editable_elements(g::Graph)
    (MinLoadedEdge(k...) for (k, e) in edges(g) if !(e.edited || e.blocked))
end

# if we need to recalculate some paths for the MinTotalDetour minmode
function can_state_change_for_min_total_detour(shortest_path_state, current_edge_iter, algorithm_config)
    if algorithm_config.buildup
        return true
    else
        # if any current edge is part of any trip, we need to recalculate
        for e in current_edge_iter
            for trip in shortest_path_state.trips
                dst = trip.destination
                while dst != trip.origin
                    src = shortest_path_state.predecessor[dst]
                    if e.source == src && e.destination == dst
                        return true
                    end
                    dst = src
                end
            end
        end
        return false
    end
end

function path_loads(state::MinTotalDetourState, shortest_path_state)
    len_before = sum(shortest_path_state.trips) do trip
        number_of_users(trip, shortest_path_state) * shortest_path_state.felt_length[trip.destination]
    end
    detours = spzeros(size(state.aggregated_detours))
    # deepcopy graph for parallel processing
    g_new = deepcopy(state.g)

    # for each editable element: edit, solve, calculate detour, edit back
    for element in editable_elements(g_new)
        if can_state_change_for_min_total_detour(shortest_path_state, to_edge_iter(element, g_new), state.algorithm_config)
            num_elem = 0  # to evenly distribute the detour across all edges in segment

            # edit all edges in current element
            for e in to_edge_iter(element, g_new)
                edit_edge!(e)
                num_elem += 1
            end

            # solve shortest paths for edited graph
            sps_new = ShortestPathState(g_new, shortest_path_state.trips)
            dynamic_vars = build_dynamic_variables(sps_new)
            solve_shortest_paths!(sps_new, dynamic_vars..., g_new)

            # calculate detour for each trip
            len_after = sum(sps_new.trips) do trip
                number_of_users(trip, sps_new) * sps_new.felt_length[trip.destination]
            end

            # set detour for each edge in element and reset element
            for e in to_edge_iter(element, g_new)
                detours[e.source, e.destination] += abs(len_after - len_before) / num_elem
                edit_edge_back!(e)
            end
        end
    end
    return detours
end

function set_path_loads!(state::MinTotalDetourState, path_id, path_detours)
    state.aggregated_detours .-= state.individual_detours[path_id]
    state.aggregated_detours .+= path_detours

    state.individual_detours[path_id] = path_detours
    return state
end

struct FixedOrderMinmodeState{T} <: AbstractMinmodeState
    aggregated_loads::SparseMatrixCSC{Float64,Int64}
    individual_loads::Vector{SparseMatrixCSC{Float64,Int64}}
    order::Vector{T}
end

function init_minmode_state(algorithm_config::AlgorithmConfig{FixedOrderMinmode{T}}, shortest_path_states, ::Graph) where T
    num_nodes = length(shortest_path_states[1].felt_length)
    aggregated_loads = spzeros(num_nodes, num_nodes)
    individual_loads = [spzeros(num_nodes, num_nodes) for _ in shortest_path_states]

    minmode_state = FixedOrderMinmodeState(aggregated_loads, individual_loads, algorithm_config.minmode.order)

    for (i, shortest_path_state) in enumerate(shortest_path_states)
        loads = path_loads(minmode_state, shortest_path_state)
        set_path_loads!(minmode_state, i, loads)
    end
    return minmode_state
end

function path_loads(state::FixedOrderMinmodeState, shortest_path_state)
    loads = spzeros(size(state.aggregated_loads))
    for trip in shortest_path_state.trips
        dst = trip.destination
        while dst != trip.origin
            src = shortest_path_state.predecessor[dst]
            loads[src, dst] += number_of_users(trip, shortest_path_state)
            dst = src
        end
    end
    return loads
end

function set_path_loads!(state::FixedOrderMinmodeState, path_id, path_loads)
    state.aggregated_loads .-= state.individual_loads[path_id]
    state.aggregated_loads .+= path_loads

    state.individual_loads[path_id] = path_loads
    return state
end