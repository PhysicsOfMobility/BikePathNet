struct Graph{T}
    nodes :: Dict{T, Node{T}}
    adjacency_lists :: Dict{T, Set{T}}
    edges :: Dict{Tuple{T,T}, Edge{T}}
end

function Graph{T}() where T
    return Graph{T}(Dict(), Dict(), Dict())
end


function Graph()
    return Graph{Int}()
end

"""
    add_node!(G, u, size)

Add node u to graph G.
"""
function add_node!(G::Graph{T}, u::T, d) where T
    G.nodes[u] = Node{T}(u, 
    d["intersection_size"],
    d["intersection_penalty"],
    )
end


"""
    remove_node!(G, u)

Remove node u from graph G.
"""
function remove_node!(G::Graph{T}, u::T) where T
    for v in get!(G.adjacency_lists, u, Set{T}())
        remove_edge!(G, u, v)
        if u in G.adjacency_lists[v]
            remove_edge!(G, v, u)
        end
    end
    delete!(G.nodes, u)
end


"""
    get_adjacent_nodes(G, u)

Return a set of adjacent nodes to u in graph G.
"""
function get_adjacent_nodes(G::Graph{T}, u::T)::Set{T} where T
    return get(G.adjacency_lists, u, Set{T}())
end


"""
    get_intersection_penalty(G, u; cyclist_type=1)

Get intersection penalty based on cyclist_type of edge (u, v) in graph G.
"""
function get_intersection_penalty(G::Graph{T}, u::T; cyclist_type=1::Int) where T
    return G.nodes[u].intersection_penalty[cyclist_type]
end


"""
    add_edge!(G, u, v, d; u_size, v_size)

Add edge (u, v) with data d to graph G.
"""
function add_edge!(G::Graph{T}, u::T, v::T, d, u_info=Dict(), v_info=Dict()) where T
    if !(haskey(G.nodes, u))
        add_node!(G, u, u_info)
    end
    if !(haskey(G.nodes, v))
        add_node!(G, v, v_info)
    end
    if !(haskey(G.adjacency_lists, u))
        G.adjacency_lists[u] = Set()
    end
    push!(G.adjacency_lists[u], v)
    G.edges[(u,v)] = Edge{T}(u, v, 
        d["real_length"],
        d["real_length"],
        d["speed"],
        d["street_type"],
        d["car_penalty"],
        d["slope"],
        d["slope_penalty"],
        d["surface"],
        d["surface_penalty"],
        d["turn_penalty"],
        d["bp_end_penalty"],
        d["ex_inf"],
        d["blocked"],
        d["bike_path"],
        d["cost"],
        d["load"],
        d["trips"],
        d["edited"]
        )
end


"""
    remove_edge!(G, u, v)

Remove edge (u, v) from graph G.
"""
function remove_edge!(G::Graph{T}, u::T, v::T) where T
    delete!(G.edges, (u, v))
    delete!(G.adjacency_lists[u], v)
end


"""
    get_edge_weight(G, u, v)

Return edge (u, v) weight from graph G.
"""
function get_edge_weight(G::Graph{T}, u::T, v::T)::Float64 where T
    return G.edges[(u, v)].weight
end


"""
    set_edge_weight!(G, u, v)

Set weight of edge (u, v) in graph G.
"""
function set_edge_weight!(G::Graph{T}, u::T, v::T, w::Float64) where T
    G.edges[(u, v)].weight = w
end


"""
    get_edge_speed(G, u, v)

Return speed based on cyclist_type of edge (u, v) from graph G.
"""
function get_edge_speed(G::Graph{T}, u::T, v::T; cyclist_type::Int=1)::Float64 where T
    return G.edges[(u, v)].speed[cyclist_type]
end


"""
    set_car_penalty!(G, u, v, p)

Set slope penalty p to edge (u, v) in graph G.
"""
function set_car_penalty!(G::Graph{T}, u::T, v::T, p::Vector{Float64}) where T
    G.edges[(u, v)].car_penalty = p
end


"""
    get_car_penalty(G, u, v; check_bp=false, cyclist_type=1)

Get car penalty based on cyclist_type of edge (u, v) in graph G.
"""
function get_car_penalty(G::Graph{T}, u::T, v::T; check_bp=false::Bool, cyclist_type::Int=1)::Float64 where T
    if (check_bp && G.edges[(u, v)].bike_path)
        return 1.0
    else
        return G.edges[(u, v)].car_penalty[cyclist_type]
    end
end


"""
    set_slope_penalty!(G, u, v, p)

Set slope penalty p to edge (u, v) in graph G.
"""
function set_slope_penalty!(G::Graph{T}, u::T, v::T, p::Vector{Float64}) where T
    G.edges[(u, v)].slope_penalty = p
end


"""
    get_slope_penalty(G}, u, v; cyclist_type=1)  

Get slope penalty based on cyclist_type of edge (u, v) in graph G.
"""
function get_slope_penalty(G::Graph{T}, u::T, v::T; cyclist_type::Int=1)::Float64 where T
    return G.edges[(u, v)].slope_penalty[cyclist_type]
end



"""
    set_turn_penalty!(G, u, v, p)

Set turn penalty p to edge (u, v) in graph G.
"""
function set_turn_penalty!(G::Graph{T}, u::T, v::T, p::Dict{T, Vector{Float64}}) where T
    G.edges[(u, v)].turn_penalty = p
end


"""
    get_turn_penalty(G, u, v, n; cyclist_type=1)

Return turn penalty p based on cyclist_type of edge (u, v) coming from node n in graph G.
"""
function get_turn_penalty(G::Graph{T}, u::T, v::T, n::T; cyclist_type::Int=1)::Float64 where T
    return get(G.edges[(u, v)].turn_penalty, n, zeros(cyclist_type))[cyclist_type]
end


"""
    get_bp_end_penalty(G::Graph{T}, u::T, v::T, on_bike_path::Bool, bike_path_ends::Int; max_end_counter::Int=3, cyclist_type=1)

Return bike path end penalty based on cyclist_type for edge (u, v) coming from node n in graph G.
"""
function get_bp_end_penalty(G::Graph{T}, u::T, v::T, on_bike_path::Bool, bike_path_ends::Int; max_end_counter::Int=3, cyclist_type::Int=1)::Float64 where T
    if (!G.edges[(u, v)].bike_path && on_bike_path)
        return get(G.edges[(u, v)].bp_end_penalty, min(bike_path_ends, max_end_counter), ones(cyclist_type))[cyclist_type]
    else
        return 0.0
    end
end
