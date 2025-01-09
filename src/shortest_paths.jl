# MARK: Aggregators
"""
    reset_aggregator!(aggregator::AbstractAggregator, source)

Resets the aggregator to the state after construction for the given source.
This allows us to reuse the allocated memory for the aggregator, instead of
recreating it for every resolve of a [`ShortestPathState`](@ref).
(The aggregator after this call should be in the same state as the return
value of `AbstractAggregator(g, source)`.)
"""
function reset_aggregator! end

"""
    aggregate_dijkstra_step!(aggregator::AbstractAggregator, state::ShortestPathState, current_edge, current_node)

Progressively fills in the aggregator as we solve the shortest path. This function
is called on the `aggregator` after every relaxation step in the Dijkstra algorithm.
"""
function aggregate_dijkstra_step! end


"""
Shortest Path Aggregator tracking if the trip to node `i` contains a street without bike path.
(If `on_street[i] == true`, the trip contains a street without bike path.)
"""
struct OnStreetAggregator <: AbstractAggregator
    on_street::Vector{Bool}
end

"""
Interface constructor for [`OnStreetAggregator`](@ref).
"""
function OnStreetAggregator(g::AbstractGraph, source)
    return OnStreetAggregator(zeros(Bool, nv(g)))
end

function reset_aggregator!(aggregator::OnStreetAggregator, source)
    fill!(aggregator.on_street, false)
    return aggregator
end

"""
Shortest Path Aggregator tracking the real distances traveled on different street types.
"""
struct RealDistanceAggregator <: AbstractAggregator
    on_all::Vector{Float64}
    on_primary::Vector{Float64}
    on_secondary::Vector{Float64}
    on_tertiary::Vector{Float64}
    on_residential::Vector{Float64}
    on_bike_path::Vector{Float64}
end

"""
Interface constructor for [`RealDistanceAggregator`](@ref).
"""
function RealDistanceAggregator(g::AbstractGraph, source)
    rda = RealDistanceAggregator((fill(Inf, nv(g)) for i in 1:6)...)
    rda.on_all[source] = 0.0

    rda.on_primary[source] = 0.0
    rda.on_secondary[source] = 0.0
    rda.on_tertiary[source] = 0.0
    rda.on_residential[source] = 0.0
    rda.on_bike_path[source] = 0.0
    return rda
end

function reset_aggregator!(rda::RealDistanceAggregator, source)
    fill!(rda.on_all, Inf)
    rda.on_all[source] = 0.0

    fill!(rda.on_primary, Inf)
    rda.on_primary[source] = 0.0
    fill!(rda.on_secondary, Inf)
    rda.on_secondary[source] = 0.0
    fill!(rda.on_tertiary, Inf)
    rda.on_tertiary[source] = 0.0
    fill!(rda.on_residential, Inf)
    rda.on_residential[source] = 0.0
    fill!(rda.on_bike_path, Inf)
    rda.on_bike_path[source] = 0.0
    return rda
end

"""
Copies the real distances in the [`RealDistanceAggregator`](@ref) to node `previous`
to node `destination`.
"""
function copy_previous_distances!(rda::RealDistanceAggregator, previous, destination)
    rda.on_all[destination] = rda.on_all[previous]

    rda.on_primary[destination] = rda.on_primary[previous]
    rda.on_secondary[destination] = rda.on_secondary[previous]
    rda.on_tertiary[destination] = rda.on_tertiary[previous]
    rda.on_residential[destination] = rda.on_residential[previous]
    rda.on_bike_path[destination] = rda.on_bike_path[previous]
    return rda
end


"""
Shortest Path Aggregator tracking no additional data. (Use this if you only need the shortest path.)
"""
struct EmptyDataAggregator <: AbstractAggregator end
"Interface constructor for [`EmptyDataAggregator`](@ref)."
EmptyDataAggregator(g, source) = EmptyDataAggregator()
reset_aggregator!(aggregator::EmptyDataAggregator, source) = aggregator


"""
Shortest Path Aggregator combining multiple aggregators. To use it, pass a
vector of aggregator-types to the `aggregator` keyword of [`ShortestPathState`](@ref).
!!! warning "Not type-stable"
    This aggregator is not type-stable, as it can (should) contain aggregators
    of different types. This allows for more flexibility when testing different
    aggregators, but can lead to performance issues if you calculate many shortest paths.
    If you want to combine mutliple aggregators, consider creating a new aggregator,
    similar to the [`EverythingAggregator`](@ref).
"""
struct CombinedAggregator <: AbstractAggregator
    aggregators::Vector{<:AbstractAggregator}
end


"""
Shortest Path Aggregator combining the [`OnStreetAggregator`](@ref) and the [`RealDistanceAggregator`](@ref).
"""
struct EverythingAggregator <: AbstractAggregator
    on_street::OnStreetAggregator
    real_distance::RealDistanceAggregator
end
"Interface constructor for [`EverythingAggregator`](@ref)."
function EverythingAggregator(g::AbstractGraph, source)
    return EverythingAggregator(OnStreetAggregator(g, source), RealDistanceAggregator(g, source))
end

reset_aggregator!(aggregator::EverythingAggregator, source) = begin
    reset_aggregator!(aggregator.on_street, source)
    reset_aggregator!(aggregator.real_distance, source)
    return aggregator
end


# MARK: ShortestPathState
"""
This type represents the shortest paths for a given set of `trips` with the same origins,
taken by the same `cyclist`.
"""
struct ShortestPathState{Agg,T}
    "trips for which we want to find the shortest paths"
    trips::Vector{T}
    "destinations contained in the trips"
    destinations::Set{Int}
    "cyclist taking the trips"
    cyclist::Cyclist
    "predescessor of each node in the shortest path (`predecessor[i]` is the node which leads to node `i` along the shortest path)"
    predecessor::Vector{Int}
    "felt length of the shortest path to each node"
    felt_length::Vector{Float64}
    "aggregator for additional data"
    data_aggregator::Agg
end

"""
    ShortestPathState(g, trips; aggregator=EmptyDataAggregator)

Constructor for a [`ShortestPathState`](@ref) from a graph `g`
and a vector of `trips` with the same `origin` and `cyclist` and different `destination`s.
The `aggregator` keyword allows to pass a single aggregator type or a vector of aggregator types.
When passing a vector, the aggregators are combined into a [`CombinedAggregator`](@ref).
"""
function ShortestPathState(g, trips; aggregator=EmptyDataAggregator)
    @assert all(trip -> trip.origin == trips[1].origin, trips) "all trips must have the same origin"

    destinations = Set(trip.destination for trip in trips)
    @assert length(destinations) == length(trips) "the destinations of all trips must be unique"
    @assert all(trip -> trip.cyclist == trips[1].cyclist, trips) "all trips must have the same cyclist"

    aggregator_instance = if aggregator isa Vector
        CombinedAggregator([a(g, trips[1].origin) for a in aggregator])
    else
        aggregator(g, trips[1].origin)
    end

    state = ShortestPathState(
        trips,
        destinations,
        trips[1].cyclist,
        zeros(Int, nv(g)),
        fill(Inf, nv(g)),
        aggregator_instance
    )
    state.felt_length[trips[1].origin] = 0.0
    return state
end


"""
Resets a [`ShortestPathState`](@ref) to the state after construction.
This way, we can reuse the allocated memory for the state.
"""
function reset_shortest_path_state!(state)
    fill!(state.predecessor, 0.0)
    fill!(state.felt_length, Inf)
    source = state.trips[1].origin
    state.felt_length[source] = 0.0
    reset_aggregator!(state.data_aggregator, source)
    return state
end

# MARK: Step-Aggregation
function aggregate_dijkstra_step!(
    aggregator::OnStreetAggregator,
    state::ShortestPathState,
    current_edge,
    current_node
)
    on_street = aggregator.on_street
    u = current_edge.source
    v = current_edge.destination

    on_street[v] = on_street[u] || !current_edge.bike_path
    return state
end

function aggregate_dijkstra_step!(
    aggregator::RealDistanceAggregator,
    state::ShortestPathState,
    current_edge,
    current_node
)
    u = current_edge.source
    v = current_edge.destination

    copy_previous_distances!(aggregator, u, v)

    # set overall length
    aggregator.on_all[v] += current_edge.length

    # set street type specific lengths
    if current_edge.bike_path
        aggregator.on_bike_path[v] += current_edge.length
    else
        # this is kind of not very elegant...
        if current_edge.street_type == :primary
            aggregator.on_primary[v] += current_edge.length
        elseif current_edge.street_type == :secondary
            aggregator.on_secondary[v] += current_edge.length
        elseif current_edge.street_type == :tertiary
            aggregator.on_tertiary[v] += current_edge.length
        elseif current_edge.street_type == :residential
            aggregator.on_residential[v] += current_edge.length
        end
    end
    return state
end

aggregate_dijkstra_step!(aggregator::EmptyDataAggregator, state::ShortestPathState, args...) = state

function aggregate_dijkstra_step!(aggregator::CombinedAggregator, state::ShortestPathState, args...)
    for sub_aggregator in aggregator.aggregators
        aggregate_dijkstra_step!(sub_aggregator, state, args...)
    end
    return state
end

function aggregate_dijkstra_step!(aggregator::EverythingAggregator, state::ShortestPathState, args...)
    aggregate_dijkstra_step!(aggregator.on_street, state, args...)
    aggregate_dijkstra_step!(aggregator.real_distance, state, args...)
    return state
end

# MARK: solve shortest paths
"""
Prepare data-structures which are only needed to run the shortest path algorithm.
Returns a tuple `(queue, visited, on_bike_path, bike_path_ends)`.
"""
function build_dynamic_variables(state)
    queue = PriorityQueue{Int,Float64}()
    queue[state.trips[1].origin] = 0.0

    n_nodes = length(state.felt_length)
    visited = zeros(Bool, n_nodes)
    on_bike_path = zeros(Bool, n_nodes)
    bike_path_ends = zeros(Int, n_nodes)

    return queue, visited, on_bike_path, bike_path_ends
end

"""
    shortest_paths(g::AbstractGraph, origin, cyclist::Cyclist; aggregator=EmptyDataAggregator)
    shortest_paths(g::AbstractGraph, origin, destination, cyclist::Cyclist; aggregator=EmptyDataAggregator)
    shortest_paths(g::AbstractGraph, origin, destinations::Set, cyclist::Cyclist; aggregator=EmptyDataAggregator)

Convenience functions to solve shortest path(s) on a graph `g` for a given `cyclist` starting at `origin`. 
"""
function shortest_paths end

function shortest_paths(g::AbstractGraph, origin, cyclist::Cyclist; aggregator=EmptyDataAggregator)
    shortest_paths(g, origin, Set(1:nv(g)), cyclist; aggregator=aggregator)
end
function shortest_paths(g::AbstractGraph, origin, destination, cyclist::Cyclist; aggregator=EmptyDataAggregator)
    shortest_paths(g, origin, Set(destination), cyclist; aggregator=aggregator)
end

function shortest_paths(g::AbstractGraph, origin, destinations::Set, cyclist::Cyclist; aggregator=EmptyDataAggregator)
    trips = [Trip(origin, dest, 1.0, cyclist, 0.0) for dest in destinations]

    # allocate all datastructures needed to solve shortest paths
    state = ShortestPathState(g, trips; aggregator=aggregator)

    dynamic_vars = build_dynamic_variables(state)

    # solve shortest paths
    solve_shortest_paths!(state, dynamic_vars..., g)
end


# this thing should at best be allocation free
"""
Solves the [`ShortestPathState`](@ref) in `state` on graph `g` using the Dijkstra algorithm.
Needs to be provided with the `queue`, `visited`, `on_bike_path`, and `bike_path_ends`.
(See [`build_dynamic_variables`](@ref).)
"""
function solve_shortest_paths!(state::ShortestPathState, queue, visited, on_bike_path, bike_path_ends, g)
    num_dests_found = 0

    while !isempty(queue)
        u = dequeue!(queue)

        # check if node we just found is the destination of any given trip
        if u in state.destinations
            num_dests_found += 1
            if num_dests_found == length(state.destinations)
                break
            end
        end

        # visit all neighbours of u
        for v in neighbours(g, u)
            visited[v] && continue
            current_edge = edges(g)[(u, v)]
            current_node = nodes(g)[v]

            speed_penalty = slope_penalty(current_edge, state.cyclist) * car_penalty(current_edge, state.cyclist; check_bp=true)

            ip = intersection_penalty(current_node, state.cyclist)
            tp = turn_penalty(current_edge, state.cyclist, state.predecessor[u])
            bpep = bike_path_end_penalty(current_edge, state.cyclist, on_bike_path[u], bike_path_ends[u])
            etp = edge_time_penalty(current_edge, state.cyclist)
            time_penalty = ip + tp + bpep + etp

            tentative_felt_length = state.felt_length[u] + (weight(current_edge) / (speed_penalty * state.cyclist.speed)) + time_penalty
            if tentative_felt_length < state.felt_length[v]
                state.felt_length[v] = tentative_felt_length

                on_bike_path[v] = current_edge.bike_path

                if on_bike_path[u] && !on_bike_path[v]
                    bike_path_ends[v] = bike_path_ends[u] + 1
                else
                    bike_path_ends[v] = bike_path_ends[u]
                end

                state.predecessor[v] = u
                # dispatch reasons
                aggregate_dijkstra_step!(state.data_aggregator, state, current_edge, current_node)

                queue[v] = tentative_felt_length
            end
        end
        visited[u] = true
    end
    return state
end


# MARK: utility functions
"""
Get the nodes of the shortest path to `destination` in the solved `state`.
"""
function nodes_from_path(state, destination)
    nodes = Int[destination]
    while state.predecessor[nodes[end]] != 0
        push!(nodes, state.predecessor[nodes[end]])
    end
    return reverse!(nodes)
end

"""
Get the edges of the shortest path to `destination` in the solved `state`.
"""
function edges_from_path(state, destination)
    edges = Tuple{Int,Int}[]
    if state.predecessor[destination] != 0
        d = destination
        s = state.predecessor[d]
        push!(edges, (s, d))
        while state.predecessor[s] != 0
            push!(edges, (state.predecessor[s], s))
            d = s
            s = state.predecessor[d]
        end
    end
    return reverse!(edges)
end

"""
Get the total felt distance travelled per trip in the solved `state`.
"""
function tripwise_total_felt_distance_traveled(state)
    return Dict(trip.destination => number_of_users(trip, state) * state.felt_length[trip.destination] for trip in state.trips)
end

"""
Get the total felt distance travelled for all trips in the solved `state`.
"""
function total_felt_distance_traveled(state)
    return sum(number_of_users(trip, state) * state.felt_length[trip.destination] for trip in state.trips)
end

"""
Get the utility for all trips in the solved shortest path `state`.
"""
function utility(state)
    return sum(trip_utility(trip, state) for trip in state.trips)
end

"""
Get the consumer surplus for all trips in the solved `state`.
"""
function consumer_surplus(state)
    return sum(0.5 * (trip.number_of_users + number_of_users(trip, state)) * (trip.t_0 - state.felt_length[trip.destination]) for trip in state.trips)
end

"""
Get the real distance travelled for all trips in the solved `state`, filtered by the
`street_type` which can one of `[:all, :primary, :secondary, :tertiary, :residential, :bike_path]` (default is `:all`).
Assumes that `state.data_aggregator` is an [`EverythingAggregator`](@ref).
"""
function real_distance_traveled_on(state, street_type=:all)
    # assuming everything aggregator
    real_distances = if street_type == :all
        state.data_aggregator.real_distance.on_all
    elseif street_type == :primary
        state.data_aggregator.real_distance.on_primary
    elseif street_type == :secondary
        state.data_aggregator.real_distance.on_secondary
    elseif street_type == :tertiary
        state.data_aggregator.real_distance.on_tertiary
    elseif street_type == :residential
        state.data_aggregator.real_distance.on_residential
    elseif street_type == :bike_path
        state.data_aggregator.real_distance.on_bike_path
    else
        throw(ArgumentError("street_type must be one of :all, :primary, :secondary, :tertiary, :residential, :bike_path"))
    end
    sum(number_of_users(trip, state) * real_distances[trip.destination] for trip in state.trips)
end


"""
Get the number of users which had to travel on the street at some point of their journey in the solved `state`.
Assumes that `state.data_aggregator` is an [`EverythingAggregator`](@ref).
"""
function number_on_street(state)
    on_street = state.data_aggregator.on_street.on_street  # everything aggregator
    sum(number_of_users(trip, state) * on_street[trip.destination] for trip in state.trips)
end

"Returns the length of an edge, with the caveat that edited bike highways have a length of `Inf`."
weight(edge) = edge.edited && edge.bike_highway ? Inf : edge.length

"""
Creates a vector of solved shortest path states from a graph `g` and a dictionary of `trips`,
indexed by their origin.
"""
function create_shortest_path_states(g, trips::Dict{Int,Vector{T}}; aggregator=EverythingAggregator) where {T}
    shortest_path_states = ShortestPathState{aggregator,T}[]
    for (origin, same_origin_trips) in trips
        for cyclist_id in unique(trip.cyclist.id for trip in same_origin_trips)
            same_cyclist_trips = [trip for trip in same_origin_trips if trip.cyclist.id == cyclist_id]

            push!(shortest_path_states, ShortestPathState(g, same_cyclist_trips, aggregator=aggregator))
        end
    end
    Threads.@threads for shortest_path_state in shortest_path_states
        dynamic_vars = build_dynamic_variables(shortest_path_state)
        solve_shortest_paths!(shortest_path_state, dynamic_vars..., g)
    end
    return shortest_path_states
end