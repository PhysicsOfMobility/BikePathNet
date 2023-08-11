using DataStructures

struct ShortestPath{T}
    dist::Dict{T,Float64}
    real_dist::Dict{T,Float64}
    real_dist_on_types::Dict{T,Dict{String,Float64}}
    on_street::Dict{T,Bool}
    nodes::Dict{T,Vector{T}}
    edges::Dict{T,Vector{Tuple{T, T}}}
end


"""
    get_distance(R, u)

Given ShortestPath R and node u, find distance. If u is not found from R,
return Inf.
"""
function get_distance(R, u)::Float64
    return get(R.dist, u, Inf)
end


"""
    get_real_distance(R, u)

Given ShortestPath R and node u, find distance. If u is not found from R,
return Inf.
"""
function get_real_distance(R, u)::Float64
    return get(R.real_dist, u, Inf)
end


"""
    get_real_distance_on_types(R, u)

Given ShortestPath R and node u, find distance. If u is not found from R,
return Inf.
"""
function get_real_distance_on_types(R, u)::Dict{String, Float64}
    return Dict{String, Float64}([(st, st_l) for (st, st_l) in get(R.real_dist_on_types, u, Dict{String, Float64}("primary" => Inf, "secondary" => Inf, "tertiary" => Inf, "residential" => Inf, "bike_path" => Inf))])
end


"""
    set_real_distance!(R, G, u, v)

Given ShortestPath R and node u, set distance.
"""
function set_real_distance!(R, G, u, v)
    R.real_dist[v] = get_real_distance(R, u) + get_edge_weight(G, u, v)
    R.real_dist_on_types[v] = get_real_distance_on_types(R, u)
    if G.edges[(u, v)].bike_path
        R.real_dist_on_types[v]["bike_path"] += get_edge_weight(G, u, v)
    else
        R.real_dist_on_types[v][G.edges[(u, v)].street_type] += get_edge_weight(G, u, v)
    end
end


"""
    get_node_sequence(prev, s)

Return node sequence of shortest paths from origin s.
"""
function get_node_sequence(prev::Dict{T, T}, s::T)::Dict{T, Vector{T}} where T
    path_nodes = Dict{T, Vector{T}}()
    for (d, p) in prev
        if d != s
            path_nodes[d] = Vector{T}()
            push!(path_nodes[d], d)
            while p != s
                push!(path_nodes[d], p)
                p = prev[p]
            end
            push!(path_nodes[d], p)
        end
    end
    return Dict{T, Vector{T}}([(d, reverse(p)) for (d, p) in path_nodes])
end


"""
    get_edge_sequence(prev, s)

Return edge sequence of shortest paths from origin s.
"""
function get_edge_sequence(prev::Dict{T, T}, s::T)::Dict{T,  Vector{Tuple{T, T}}} where T
    path_edges = Dict{T, Vector{Tuple{T, T}}}()
    for (d, p) in prev
        if d != s
            path_edges[d] = Vector{T}()
            u = d
            while p != s
                push!(path_edges[d], (u, p))
                u = p
                p = prev[p]
            end
            push!(path_edges[d], (u, p))
        end
    end
    return Dict{T, Vector{Tuple{T, T}}}([(d, [reverse(e) for e in reverse(p)]) for (d, p) in path_edges])
end


"""
    ShortestPath(G::Graph{T}, s::T, cyclist_type::Int=1)::ShortestPath{T}

Find shortest paths in graph G, starting from node s.
"""
function ShortestPath(G::Graph{T}, s::T; cyclist_type::Int=1)::ShortestPath{T} where T
    R = ShortestPath{T}(Dict{T,Float64}(), Dict{T,Float64}(), Dict{T,Dict{String, Float64}}(), Dict{T,Bool}(), Dict{T, Vector{T}}(), Dict{T, Vector{Tuple{T, T}}}())
    visited = Set{T}()
    queue = PriorityQueue{T,Float64}()
    dist = R.dist
    dist[s] = 0.0
    real_dist = R.real_dist
    real_dist[s] = 0.0
    real_dist_on_types = R.real_dist_on_types
    real_dist_on_types[s] = Dict{String, Float64}("primary" => 0.0, "secondary" => 0.0, "tertiary" => 0.0, "residential" => 0.0, "bike_path" => 0.0)
    on_street = R.on_street
    on_street[s] = false
    prev = Dict{T, T}()
    prev[s] = s
    queue[s]= 0.0
    bike_path_ends = Dict{T, Int}()
    bike_path_ends[s] = 0
    on_bike_path = Dict{T, Bool}()
    on_bike_path[s] = false
    while (!isempty(queue))
        u = dequeue!(queue)
        for v in get_adjacent_nodes(G, u)
            v in visited && continue
            speed_penalty =  get_slope_penalty(G, u, v, cyclist_type=cyclist_type) * get_car_penalty(G, u, v, check_bp=true, cyclist_type=cyclist_type)
            time_penalty = get_intersection_penalty(G, v, cyclist_type=cyclist_type) + get_turn_penalty(G, u, v, prev[u], cyclist_type=cyclist_type) + get_bp_end_penalty(G, u, v, on_bike_path[u], bike_path_ends[u], cyclist_type=cyclist_type)

            tentative_distance = get_distance(R, u) + (get_edge_weight(G, u, v) / (speed_penalty * get_edge_speed(G, u, v, cyclist_type=cyclist_type))) + time_penalty
            if tentative_distance < get_distance(R, v)
                dist[v] = tentative_distance
                set_real_distance!(R, G, u, v)
                if on_street[u] || !G.edges[(u, v)].bike_path
                    on_street[v] = true
                else
                    on_street[v] = false
                end
                on_bike_path[v] = G.edges[(u, v)].bike_path
                if on_bike_path[prev[u]] && !on_bike_path[v]
                        bike_path_ends[v] = bike_path_ends[u] + 1
                else
                        bike_path_ends[v] = bike_path_ends[u]
                end

                prev[v] = u
                queue[v] = tentative_distance
            end
        end
        push!(visited, u)
    end
    return ShortestPath(dist, real_dist, real_dist_on_types, on_street, get_node_sequence(prev, s), get_edge_sequence(prev, s))
end
