"""
Returns the fraction, by length, of streets in network 'G' which have a bike
path, excluding ramps and blocked streets.
"""
function bike_path_percentage(G)
    bike_length = 0
    total_length = 0
    for (edge, edge_info) in edges(G)
        if !edge_info.ramp && !edge_info.blocked
            total_length += edge_info.length
            if edge_info.bike_path
                bike_length += edge_info.length
            end
        end
    end
    return bike_length / total_length
end

"""
Returns the cumulated cost if all edges in `bike_paths` where to receive a bike path in graph `g`.
If `ex_inf==true`, edges with `ex_inf==true` are excluded.
"""
function total_cost(g, bike_paths, ex_inf)
    total_cost = 0.0
    for k in bike_paths
        edge = edges(g)[k]
        if !ex_inf || !edge.ex_inf
            total_cost += edge.cost
        end
    end
    return total_cost
end