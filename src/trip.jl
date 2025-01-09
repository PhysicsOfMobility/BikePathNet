"""
A trip from `origin` to `destination`, taken by `number_of_users` cyclists with a profile
specified by a [`Cyclist`](@ref).
"""
struct Trip
    "`id` of origin [`Node`](@ref)"
    origin::Int
    "`id` of destination [`Node`](@ref)"
    destination::Int
    "number of users taking the trip"
    number_of_users::Float64
    "profile of the cyclists taking the trip"
    cyclist::Cyclist
    "time traveled in the base state"
    t_0::Float64
end

number_of_users(t::Trip, _) = t.number_of_users
trip_utility(trip::Trip, shortest_path_state) = -trip.number_of_users * shortest_path_state.felt_length[trip.destination]
trip_consumer_surplus(trip::Trip, shortest_path_state) = trip.number_of_users * (trip.t_0 - shortest_path_state.felt_length[trip.destination])

"""
A trip from `origin` to `destination`, taken by at least `number_of_users` cyclists with a profile
specified by a [`Cyclist`](@ref). The actual number of users changes, depending on
the current felt length of the trip compared to the felt length of the trip in the state without bike paths.
(see TODO: add reference to paper once published).
"""
struct VariableTrip
    "`id` of origin [`Node`](@ref)"
    origin::Int
    "`id` of destination [`Node`](@ref)"
    destination::Int
    "minimum number of users taking the trip"
    number_of_users::Float64
    "profile of the cyclists taking the trip"
    cyclist::Cyclist
    "time traveled in the base state"
    t_0::Float64
    "distance traveled in the base state"
    d_0::Float64
    "marginal utility of travel time for bikes"
    beta::Float64
    "probability (TODO: not really a probability...) of taking the trip by bike given no bike paths"
    p_bike_0::Float64
    "probability (TODO: not really a probability...) of taking the trip by any other mode than by bike"
    p_other::Float64
    "value of time (price per unit time)"
    vot::Float64
end

"""
Number of users taking the trip `trip` given the felt lengths in `shortest_path_state`. (if `trip::Trip` the `shortest_path_state` is ignored.)
"""
function number_of_users(trip::VariableTrip, shortest_path_state)   # this trip should in general already be in shortest_path_state.trips...
    p_t = exp(trip.beta * shortest_path_state.felt_length[trip.destination])
    p_0 = trip.p_bike_0
    return trip.number_of_users * (p_t / (p_t + trip.p_other)) / (p_0 / (p_0 + trip.p_other))
end

"""
Utility of the trip `trip` given the felt lengths in `shortest_path_state`. (if `trip::Trip` it is equivalent to the total felt length of the trip.)
"""
function trip_utility(trip::VariableTrip, shortest_path_state)
    return  - trip.number_of_users * log((exp(trip.beta * shortest_path_state.felt_length[trip.destination]) + trip.p_other) / (trip.p_bike_0 + trip.p_other)) / ((trip.p_bike_0 / (trip.p_bike_0 + trip.p_other)) * trip.beta)
end

"""
Consumer surplus of the trip `trip` given the felt lengths in `shortest_path_state`. (if `trip::Trip` rule of half is ignored.)
"""
function trip_consumer_surplus(trip::VariableTrip, shortest_path_state)
    return 0.5 * (trip.number_of_users + number_of_users(trip, shortest_path_state)) * (trip.t_0 - shortest_path_state.felt_length[trip.destination])
end

function Base.show(io::IO, ::MIME"text/plain", trip::T) where {T<:Union{Trip,VariableTrip}}
    println(io, T)
    for field in fieldnames(T)
        println(io, "  ", field, ": ", getfield(trip, field))
    end
end

function parse_trips(path::String, cyclists::Vector{Cyclist})
    @assert length(unique(c.id for c in cyclists)) == length(cyclists) "all cyclists must have unique ids"

    demand_json = JSON.parsefile(path)
    demand_json = Dict(eval(Meta.parse(k)) => Dict(parse(Int, i) => l for (i, l) in v) for (k, v) in demand_json)
    filter!(p -> p[1][1] != p[1][2], demand_json)
    return demand_json
end

"""
Loads the `json` file at `path` into a `triptype`.

The json file is formatted as follows:
```json
{
    "(origin, destination)": {
        "cyclist_id1": number_of_users1,
        "cyclist_id2": number_of_users2,
        ...
    },
    ...
}
```
Where the `cyclist_id` is the `id` of the [`Cyclist`](@ref) taking the trip, given via the vector `cyclists`.
"""
function load_trips(path::String, cyclists::Vector{Cyclist}, triptype::Type{Trip})
    demand_json = parse_trips(path, cyclists)

    trips = Dict{Int,Vector{Trip}}()
    for ((origin, destination), loads) in demand_json
        if !haskey(trips, origin)
            trips[origin] = Trip[]
        end
        for (cyclist_id, load) in loads
            position = findfirst(c -> c.id == cyclist_id, cyclists)
            if !(load isa Number) 
                load = load["number_of_users"]
            end
            push!(trips[origin], Trip(origin, destination, load, cyclists[position], 0.0))
        end
    end
    return trips
end

function load_trips(path::String, cyclists::Vector{Cyclist}, ::Type{VariableTrip})
    demand_json = parse_trips(path, cyclists)

    trips = Dict{Int,Vector{VariableTrip}}()
    for ((origin, destination), demand_infos) in demand_json
        if !haskey(trips, origin)
            trips[origin] = VariableTrip[]
        end
        for (cyclist_id, demand_info) in demand_infos
            position = findfirst(c -> c.id == cyclist_id, cyclists)
            trip = VariableTrip(
                origin,
                destination,
                demand_info["number_of_users"],
                cyclists[position],
                0.0,
                0.0,
                1.0,
                1.0,
                demand_info["p_other"],
                demand_info["vot"])
            push!(trips[origin], trip)
        end
    end
    return trips
end