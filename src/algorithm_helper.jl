using HDF5
using JSON


"""
    parse_tuple(tuple_str::String, tuple_type::DataType)
Parses the string of a 2-Tuple to a Tuple of given type. 
"""
function parse_tuple(tuple_str::String, tuple_type::DataType)::Tuple{tuple_type,tuple_type}
    return Tuple{tuple_type,tuple_type}(parse.(tuple_type, split(chop(tuple_str, head = 1, tail = 1), ',')))
end


"""
    parse_dict(dict_str::String)
Parses the string of a dictionary to a Tuple of given type. 
"""
function parse_dict(dict_str::String, key_type::DataType=Int, value_type::DataType=String)::Dict{key_type,value_type}
    dict_str = chop(dict_str, head = 1, tail = 1)
    return Dict{key_type,value_type}([(parse(key_type,split(dict_str_i, ": ")[1]), chop(split(dict_str_i, ": ")[2], head = 1, tail = 1)) for dict_str_i in split(dict_str, ", ")])
end


"""
    load_algorithm_params(path::String)

Loads all relevant paramters for the complete algorithm from the file given by the 'path'.
"""
function load_algorithm_params(path::String)
    params = JSON.parsefile(path)

    mode = Tuple{Bool,Int,Bool}(params["mode"])
    speeds = Vector{Float64}(params["speeds"])
    car_penalties = Dict{String,Vector{Float64}}(params["car_penalty"])
    slope_penalties = Dict{Float64,Vector{Float64}}([(parse(Float64, k), v) for (k, v) in params["slope_penalty"]])
    surface_penalties = Dict{String,Vector{Float64}}(params["surface_penalty"])
    intersection_penalties = Dict{String,Vector{Float64}}(params["intersection_penalty"])
    turn_penalties = Dict{String,Vector{Float64}}(params["turn_penalty"])
    bp_end_penalties = Dict{String,Dict{Int,Vector{Float64}}}([(k, Dict{Int,Vector{Float64}}([(parse(Int, k_i), v_i) for (k_i, v_i) in v])) for (k, v) in params["bp_end_penalty"]])
    save_edge_load = params["save_edge_load"]
    cyclist_types = params["cyclist_types"]
    blocked = params["blocked"]

    return mode, speeds, car_penalties, slope_penalties, surface_penalties, intersection_penalties, turn_penalties, bp_end_penalties, save_edge_load, cyclist_types, blocked
end


"""
    load_comparison_state_params(path::String)

Loads all relevant paramters for the comparison state calculation from the file given by the 'path'.
"""
function load_comparison_state_params(path::String)
    params = JSON.parsefile(path)

    speeds = Vector{Float64}(params["speeds"])
    car_penalties = Dict{String,Vector{Float64}}(params["car_penalty"])
    slope_penalties = Dict{Float64,Vector{Float64}}([(parse(Float64, k), v) for (k, v) in params["slope_penalty"]])
    surface_penalties = Dict{String,Vector{Float64}}(params["surface_penalty"])
    intersection_penalties = Dict{String,Vector{Float64}}(params["intersection_penalty"])
    turn_penalties = Dict{String,Vector{Float64}}(params["turn_penalty"])
    bp_end_penalties = Dict{String,Dict{Int,Vector{Float64}}}([(k, Dict{Int,Vector{Float64}}([(parse(Int, k_i), v_i) for (k_i, v_i) in v])) for (k, v) in params["bp_end_penalty"]])
    bike_paths = Set{Tuple{Int,Int}}([(edge[1], edge[2]) for edge in params["bike_paths"]])
    ex_inf = Bool(params["ex_inf"])
    cyclist_types = params["cyclist_types"]
    blocked = params["blocked"]

    return speeds, car_penalties, slope_penalties, surface_penalties, intersection_penalties, turn_penalties, bp_end_penalties, bike_paths, ex_inf, cyclist_types, blocked
end


"""
    load_graph(path::String, car_penalties::Dict{String,Vector{Float64}}, slope_penalties::Dict{Float64,Vector{Float64}}, intersection_penalties::Dict{String, Vector{Float64}}, turn_penalties::Dict{String, Vector{Float64}}, bp_end_penalties::Dict{String,Dict{Int,Vector{Float64}}},
    ex_inf::Bool=false, blocked::Bool=false, buildup::Bool=false, cyclist_types::Int=1)

Loads the graph from the given 'path'. Sets penalties according to the given penalty dictionaries. If existing infrastructure should be set, set 'ex_inf' to true.
If the algorithm should start fro scratch set 'buildup' to false.
"""
function load_graph(path::String, speeds::Vector{Float64}, car_penalties::Dict{String,Vector{Float64}}, slope_penalties::Dict{Float64,Vector{Float64}}, surface_penalties::Dict{String,Vector{Float64}}, intersection_penalties::Dict{String, Vector{Float64}}, turn_penalties::Dict{String, Vector{Float64}}, bp_end_penalties::Dict{String,Dict{Int,Vector{Float64}}},
    ex_inf::Bool=false, blocked::Bool=false, buildup::Bool=false, cyclist_types::Int=1)::Graph

    graph_data = JSON.parsefile(path)
    edge_data = Dict{Tuple{Int, Int}, Dict}(
        [(parse_tuple(edge, Int), Dict{String, Any}([
            ("length", edge_info["length"]),
            ("street_type", edge_info["street_type"]),
            ("cost", get(edge_info, "cost", edge_info["length"])),
            ("ex_inf", get(edge_info, "ex_inf", false)),
            ("blocked", get(edge_info, "blocked", false)),
            ("turn_penalty", Dict{Int,String}([(parse(Int,k), v) for (k, v) in edge_info["turn_penalty"]])),
            ("slope", get(edge_info, "slope", 0.0)),
            ("surface", get(edge_info ,"surface", "asphalt"))
            ])
        ) for (edge, edge_info) in graph_data["edge_data"]]
    )
    node_data = Dict{Int,Dict{String,Any}}([(parse(Int,node), node_data) for (node,node_data) in graph_data["node_data"]])

    edge_dict = Dict{Tuple{Int, Int}, Dict}(
        [(edge, Dict{String, Any}([
            ("real_length", edge_info["length"]),
            ("street_type", edge_info["street_type"]),
            ("speed", speeds),
            ("car_penalty", get(car_penalties, edge_info["street_type"], ones(cyclist_types))),
            ("slope", edge_info["slope"]),
            ("slope_penalty", get_slope_penalty_value(edge_info["slope"], slope_penalties, cyclist_types)),
            ("surface", edge_info["surface"]),
            ("surface_penalty", get(surface_penalties, edge_info["surface"], ones(cyclist_types))),
            ("turn_penalty", get_turn_penalties(Dict{Int,String}(edge_info["turn_penalty"]), turn_penalties)),
            ("bp_end_penalty", bp_end_penalties[edge_info["street_type"]]),
            ("ex_inf", false),
            ("blocked", false),
            ("bike_path", !buildup),
            ("cost", edge_info["cost"]),
            ("load", zeros(cyclist_types)),
            ("trips", Vector{Tuple{Int,Int}}()),
            ("edited", false)
            ])
        ) for (edge, edge_info) in edge_data]
    )
    node_dict = Dict{Int, Dict}(
        [(node,  Dict{String, Any}([
            ("intersection_size", node_info["intersection_size"]),
            ("intersection_penalty", get(intersection_penalties, node_info["intersection_size"], 0.0)),
            ("bike_highway", false)
            ])
        ) for (node, node_info) in node_data]  
    )

    if ex_inf
        for (edge, edge_info) in edge_dict
            edge_info["ex_inf"] = edge_data[edge]["ex_inf"]
            edge_info["bike_path"] = !buildup || edge_data[edge]["ex_inf"] 
        end
    end

    if blocked
        for (edge, edge_info) in edge_dict
            if edge_data[edge]["blocked"]
                edge_info["blocked"] = true
                edge_info["bike_path"] = false
                edge_info["edited"] = true
            end
        end
    end

    G = Graph()
    for (edge, edge_info) in edge_dict
        add_edge!(G, edge[1], edge[2], edge_info, node_dict[edge[1]], node_dict[edge[2]])
    end
    return G
end


"""
    load_demand(path, cyclist_types=1, street_type=["primary", "secondary", "tertiary", "residential", "bike_path"])

Loads the demand from the given 'path'.
"""
function load_demand(path::String, cyclist_types=1::Int, street_types=["primary", "secondary", "tertiary", "residential", "bike_path"]::Vector{String})::Tuple{Dict{Int,Trip},Set{Int}}
    demand_json = JSON.parsefile(path)
    demand = Dict{Tuple{Int,Int},Dict{Int,Float64}}([(parse_tuple(trip,Int), Dict{Int,Float64}([(parse(Int, cyclist_type), trip_nbr) for (cyclist_type,trip_nbr) in trip_nbrs])) for (trip,trip_nbrs) in demand_json])
    demand = Dict{Tuple{Int,Int},Dict{Int,Float64}}([(trip, trip_nbrs) for (trip,trip_nbrs) in demand if trip[1] != trip[2]])

    stations = Set{Int}([station for (trip_id, nbr_of_users) in demand for station in trip_id])

    trips = Dict{Int,Trip}()
    trip_id = 1
    for (od, trip_nbrs) in demand
        for cyclist_type in 1:cyclist_types
            trips[trip_id] = Trip(od[1], od[2], trip_nbrs[cyclist_type], Vector{Int}(), Vector{Tuple{Int, Int}}(), 0.0, 0.0, 
            Dict{String,Float64}([(t, 0.0) for t in street_types]), 
            false, cyclist_type)
            trip_id += 1
        end
    end

    return trips, stations
end


"""
    get_slope_penalty_value(slope::Float64, slope_penalites::Dict{Float64,Vector{Float64}}, cyclist_types)::Vector{Float64}

Returns the penalty for the given 'slope', given the penalty distribution in 'slope_penalites'.
"""
function get_slope_penalty_value(slope::Float64, slope_penalites::Dict{Float64,Vector{Float64}}, cyclist_types=1::Int)::Vector{Float64}
    if slope < 0.0
        slope_penalites = Dict{Float64,Vector{Float64}}([(k, v) for (k, v) in slope_penalites if k < 0.0])
        return [maximum([v[cyclist_type] for (k, v) in slope_penalites if slope <= k]; init=1.0) for cyclist_type in 1:cyclist_types]
    else
        return [maximum([v[cyclist_type] for (k, v) in slope_penalites if slope >= k]; init=1.0) for cyclist_type in 1:cyclist_types]
    end
end


"""
    get_turn_penalties(intersection::Dict{Int,String}, intersection_penalites::Dict{String,Vector{Float64}})::Dict{Int,Vector{Float64}}

Returns the penalty for the given 'intersection', given the penalty distribution in 'intersection_penalites'.
"""
function get_turn_penalties(intersection::Dict{Int,String}, turn_penalites::Dict{String,Vector{Float64}})::Dict{Int,Vector{Float64}}
    intersection_turn_penalites = Dict{Int, Vector{Float64}}()
    for (incoming, intersection_size) in intersection
        intersection_turn_penalites[incoming] = turn_penalites[intersection_size]
    end
    return intersection_turn_penalites
end


function get_used_primary_secondary(G::Graph)::Vector{Tuple{Int,Int}}
    p_s_list = Vector{Tuple{Int,Int}}()
    for (edge, edge_info) in G.edges
        if edge_info.bike_path && (edge_info.street_type in ["primary", "secondary"])
            push!(p_s_list, edge)
        end
    end
    return p_s_list
end


"""
    get_total_cost(G::Graph, bike_paths::Union{Vector{Tuple{Int,Int}}, Set{Tuple{Int,Int}}}, ex_inf::Bool=false)::Float64

Returns the total cost to build the given 'bike_paths' in network 'G'. 
For 'ex_inf' true, the cost for already existing bike paths is ignored, for 'ex_inf' false, the cost is also factored in.
"""
function get_total_cost(G::Graph, bike_paths::Union{Vector{Tuple{Int,Int}}, Set{Tuple{Int,Int}}}, ex_inf::Bool=false)::Float64
    total_cost = 0.0
    for edge in bike_paths
        if !ex_inf || !G.edges[edge].ex_inf
            total_cost += G.edges[edge].cost
        end
    end
    return total_cost
end


"""
    check_if_trip_on_street(trip::Trip, G::Graph)::Bool

Checks if the given 'trip' is driving at some point on a street without a bike path.
"""
function check_if_trip_on_street(trip::Trip, G::Graph)::Bool
    for edge in trip.edges
        if !G.edges[edge].bike_path
            return true
        end
    end
    return false
end


"""
    delete_edge_load!(G::Graph, trips::Dict{Int,Trip{Int}})::Graph

Deletes the load from the edges in network 'G', inflicted by the given 'trips'.
"""
function delete_edge_load!(G::Graph, trips::Dict{Int,Trip{Int}})
    for (trip_id, trip) in trips
        for edge in trip.edges
            filter!(x->x!=trip_id,G.edges[edge].trips)
            G.edges[edge].load[trip.cyclist_type] -= trip.nbr_of_users
        end
    end
end


"""
    get_undirected_edge_load(G::Graph, edited::Bool)::Dict{Tuple{Int,Int},Float64}

Returns an undirected version of the edge load dictionary for the given network 'G'.
If 'edited' is set to false, only un-edited edges are considered, if 'edited' is set to true, only edited edges.
"""
function get_undirected_edge_load(G::Graph, edited::Bool)::Dict{Tuple{Int,Int},Float64}
    edge_loads = Dict{Tuple{Int,Int},Float64}()
    for (edge, edge_info) in G.edges
        edge_rev = reverse(edge)
        if !haskey(edge_loads, edge) && !haskey(edge_loads, edge_rev)
            if edge_info.edited == edited
                edge_loads[edge] = edge_info.load + G.edges[edge_rev].load
            end
        end
    end
    return edge_loads
end


"""
    get_minimal_loaded_edge(G::Graph, trips::Dict{Int,Trip}=Dict{Int,Trip}(0 => Trip(0,0)), minmode::Int=1, buildup::Bool=false, cyclist_types::Int=1)::Tuple{Int,Int}

Returns the minimal loaded edge in network 'G'. The 'trips', is only neccessary for 'minmode' 2.
Depending on the 'minmode' different definitions of load are considered. 
    0: pure number of cyclists on an edge
    1: number of cyclists on an edge weighted by the edges street penalty
    2: detour inflicted by removing a bike path from an edge
'buildup' should be used to define if the algorithm is used in "removing" or "buildup" mode.
"""
function get_minimal_loaded_edge(G::Graph, trips::Dict{Int,Trip}=Dict{Int,Trip}(0 => Trip(0,0)), minmode::Int=1, buildup::Bool=false, cyclist_types::Int=1; undirected::Bool=false)::Tuple{Tuple{Int,Int},Bool}
    if undirected
        edge_loads = get_undirected_edge_load(G, false)
    else
        edge_loads = Dict{Tuple{Int,Int},Vector{Float64}}([(edge, edge_info.load) for (edge, edge_info) in G.edges if !edge_info.edited])
    end
    if minmode == 0
        edge_loads = Dict{Tuple{Int,Int},Float64}([(edge, sum(load)) for (edge, load) in edge_loads])
    elseif minmode == 1
        edge_loads = Dict([(edge, sum(load ./ G.edges[edge].car_penalty)) for (edge, load) in edge_loads])
    elseif minmode == 2
        if !(length(edge_loads) == 0) && !(minimum([sum(load) for load in values(edge_loads)]) == 0.0)
            edge_loads = get_edge_detour(G, [e for (e, load) in edge_loads], trips, buildup, cyclist_types, undirected=undirected)
        end
    else
        print("Minmode has to be chosen. Aborting.")
        edge_loads = Dict()
    end
    if length(edge_loads) == 0
        return (-1, -1), false
    else
        if !buildup
            min_load = minimum(collect(values(edge_loads)))
            if min_load == 0.0
                used = false
            else
                used = true
            end
            return rand([e for (e, load) in edge_loads if load == min_load]), used
        else
            max_load = maximum(collect(values(edge_loads)))
            if max_load == 0.0
                used = false
            else
                used = true
            end
            return rand([e for (e, load) in edge_loads if load == max_load]), used
        end
    end
end


"""
    get_edge_detour(G::Graph, edges::Vector{Tuple{Int,Int}}, trips::Dict{Int,Trip}, buildup::Bool=false)::Dict{Tuple{Int,Int},Float64}

Returns the total detour added for each edge given in 'edges', if the bike path would be removed from the edge. The Detour is calculated for the given 'trips'.
If 'buildup' is set true, the detour removed is calculated, if a bike path would be added to the edge.
"""
function get_edge_detour(G::Graph, edges::Vector{Tuple{Int,Int}}, trips::Dict{Int,Trip}, buildup::Bool=false, cyclist_types=1::Int; undirected::Bool=false)::Dict{Tuple{Int,Int},Float64}
    len_before = sum([trip.nbr_of_users * trip.felt_length for trip in values(trips)])
    len_after = Dict{Tuple{Int,Int},Float64}()
    for edge in edges
        trips_calc = deepcopy(trips)
        G_calc = deepcopy(G)
        edit_edge!(G_calc, edge, undirected=undirected)
        trips_recalc = Dict{Int,Trip}([(trip_id, trips_calc[trip_id]) for trip_id in G_calc.edges[edge].trips])
        calc_trips!(G_calc, trips_recalc, cyclist_types)
        trips_calc = merge(trips_calc, trips_recalc)
        len_after[edge] = sum([trip.nbr_of_users * trip.felt_length for trip in values(trips_calc)])
    end
    if !buildup
        return Dict{Tuple{Int,Int},Float64}([(edge, len - len_before) for (edge, len) in len_after])
    else
        return Dict{Tuple{Int,Int},Float64}([(edge, len_before - len) for (edge, len) in len_after])
    end
end


"""
    add_edge_load!(G::Graph, edge_load::Dict{Tuple{Int,Int},Vector{Float64}})

Returns the load of streets in network 'G' to edge_load dictionary.
"""
function add_edge_load!(G::Graph, edge_load::Dict{Tuple{Int,Int},Vector{Float64}})
    for (edge, edge_info) in G.edges
        push!(edge_load[edge], sum(edge_info.load))
    end
end


"""
    get_edge_load(G::Graph)::Dict{Tuple{Int,Int},Int}

Returns the load of streets in network 'G' to edge_load dictionary.
"""
function get_edge_load!(G::Graph)::Dict{Tuple{Int,Int},Int}
    return Dict{Tuple{Int,Int},Int}([(edge, edge_info.load) for (edge, edge_info) in G.edges])
end


"""
    bike_path_percentage(G::Graph)::Float64

Returns the fraction, by length, of streets in network 'G' which have a bike path.
"""
function bike_path_percentage(G::Graph)::Float64
    bike_length = 0
    total_length = 0
    for (edge, edge_info) in G.edges
        if !edge_info.blocked
            total_length += edge_info.real_length
            if edge_info.bike_path
                bike_length += edge_info.real_length
            end
        end
    end
    return bike_length / total_length
end


"""
    nbr_of_trips_on_street(trips::Dict{Int,Trip})::Int

Returns the total number of cyclists that have to drive at some point on a street without a bike path.
"""
function nbr_of_trips_on_street(trips::Dict{Int,Trip})::Float64
    return sum([trip.nbr_of_users for trip in values(trips) if trip.on_street])
end


"""
    real_dist_on_types(trip::Trip, G::Graph)::Dict{String,Float64}

Returns the total real length driven by the given 'trip' in network 'G', split up by the different street types in 'G'.
"""
function real_dist_on_types(trip::Trip, G::Graph)::Dict{String,Float64}
    len_on_type = Dict{String,Float64}([(t, 0.0) for (t, l) in trip.real_length_on_types])
    for edge in trip.edges
        if G.edges[edge].bike_path
            len_on_type["bike_path"] += G.edges[edge].real_length
        else
            len_on_type[G.edges[edge].street_type] += G.edges[edge].real_length
        end
    end
    return len_on_type
end


"""
    total_real_dist_on_types(trips::Dict{Int,Trip}, len_type::String="real")::Dict{String, Float64}

Returns the total distance traveled by all cyclists and also the split between the different street types in 'G'. 'len_type' defines if the felt or real distance should be considered. 
    Return dict is keyed with "total length on all", "total length on street", "total length on bike paths" and for each street type "total length on 'street type'".
"""
function total_real_dist_on_types(trips::Dict{Int,Trip})::Dict{String, Float64}
    lop = 0.0
    los = 0.0
    lot = 0.0
    lor = 0.0
    lob = 0.0
    for trip in values(trips)
        lop += trip.nbr_of_users * trip.real_length_on_types["primary"]
        los += trip.nbr_of_users * trip.real_length_on_types["secondary"]
        lot += trip.nbr_of_users * trip.real_length_on_types["tertiary"]
        lor += trip.nbr_of_users * trip.real_length_on_types["residential"]
        lob += trip.nbr_of_users * trip.real_length_on_types["bike_path"]
    end
    tlos = lop + los + lot + lor
    tloa = tlos + lob
    return Dict("total length on all" => tloa, "total length on street" => tlos,
            "total length on primary" => lop, "total length on secondary" => los,
            "total length on tertiary" => lot, "total length on residential" => lor,
            "total length on bike paths" => lob)
    
end


"""
    edit_edge!(G::Graph, edge::Tuple{Int,Int}; undirected::Bool=False)::Graph

Edits the given 'edge' in a undirected network 'G'. Inverts the bike path attribute.
"""
function edit_edge!(G::Graph, edge::Tuple{Int,Int}; undirected::Bool=false)
    G.edges[edge].edited = true
    G.edges[edge].bike_path = !G.edges[edge].bike_path
    if undirected
        edge_rev = reverse(edge)
        G.edges[edge_rev].edited = true
        G.edges[edge_rev].bike_path = !G.edges[edge_rev].bike_path
    end
end


"""
    set_sp_info!(source::Int, shortest_paths::ShortestPath, G::Graph, trips::Dict{Int,Trip})::Tuple{Graph,Dict{Int,Trip}}

Sets all relevant information to the edges of network 'G' and the 'trips', if they are affected by the given 'shortest_paths' starting from the 'source' node.
"""
function set_sp_info!(source::Int, shortest_paths::Dict{Int,ShortestPath}, G::Graph, trips::Dict{Int,Trip})
    for (trip_id, trip) in trips
        if trip.origin == source
            cyclist_type = trip.cyclist_type
            if !(trip.nodes == shortest_paths[cyclist_type].nodes[trip.destination])
                delete_edge_load!(G, Dict(trip_id => trip))
                trip.nodes = shortest_paths[cyclist_type].nodes[trip.destination]
                trip.edges = shortest_paths[cyclist_type].edges[trip.destination]
                for edge in trip.edges
                    push!(G.edges[edge].trips, trip_id)
                    G.edges[edge].load[cyclist_type] += trip.nbr_of_users
                end
            end
            trip.felt_length = shortest_paths[cyclist_type].dist[trip.destination]
            trip.real_length = shortest_paths[cyclist_type].real_dist[trip.destination]
            trip.real_length_on_types = shortest_paths[cyclist_type].real_dist_on_types[trip.destination]
            trip.on_street = shortest_paths[cyclist_type].on_street[trip.destination]
        end
    end
end


"""
    calc_trips!(G::Graph, trips::Dict{Int,Trip}, cyclist_types=1)

Calculates all shortest paths for the given 'trips' in network 'G' and sets all relevant information to the edges of network 'G' and the 'trips'.
"""
function calc_trips!(G::Graph, trips::Dict{Int,Trip}, cyclist_types=1::Int)
    for source in unique!([trip.origin for trip in values(trips)])
        sp_info = Dict{Int,ShortestPath}()
        Threads.@threads for cyclist_type in 1:cyclist_types
            sp_info[cyclist_type] = ShortestPath(G, source, cyclist_type=cyclist_type)
        end
        set_sp_info!(source, sp_info, G, trips)
    end
end