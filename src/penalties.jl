"""
Multiplicative penalty for the slope of the street.
"""
function slope_penalty(edge, cyclist)
    if edge.slope < 0.0
        return minimum(v for (k, v) in cyclist.slope_penalty if edge.slope <= k && k < 0.0; init=1.0)
    else
        return minimum(v for (k, v) in cyclist.slope_penalty if edge.slope >= k && k >= 0; init=1.0)
    end
end

"""
Multiplicative penalty for using unprotected streets. Uses the `street_type` of [`Edge`](@ref).
"""
function car_penalty(edge, cyclist; check_bp=false)
    if check_bp && edge.bike_path
        return 1.0
    else
        return cyclist.car_penalty[edge.street_type]
    end
end

"""
Additive penalty for the size of intersections. Uses the `size` of [`Node`](@ref).
"""
function intersection_penalty(vertex, cyclist)
    return cyclist.intersection_penalty[vertex.size]
end

"""
Additive penalty for the time it takes to traverse a street. Uses the `time_penalty` of [`Edge`](@ref).
"""
function edge_time_penalty(edge, cyclist)
    return cyclist.edge_time_penalty[edge.time_penalty]
end

"""
Additive penalty for turning from one edge to another. Uses the `turn_badness` of the [`Edge`](@ref) we turn on.
"""
function turn_penalty(edge, cyclist, from)
    if haskey(edge.turn_badness, from)
        return cyclist.turn_penalty[edge.turn_badness[from]]
    else
        0.0
    end
end

"""
Additive penalty for cycling of a bike path, onto a unprotected street.
Uses the `street_type` of the [`Edge`](@ref) we turn on, as well as the number of times the cyclist left a bike lane already.
"""
function bike_path_end_penalty(edge, cyclist, on_bike_path, bike_path_ends)
    if !edge.bike_path && on_bike_path
        penalty_dict = cyclist.bikepath_end_penalty[edge.street_type]
        final_key = min(bike_path_ends, length(penalty_dict))
        return get(penalty_dict, final_key, 0.0)
    else
        return 0.0
    end
end

# simpler access to the penalties
"""
Returns a named tuple containing all the additive time penalties.
"""
function time_penalties(current_edge, current_node, cyclist; predecessor=-1, on_bike_path=false, bike_path_ends=0)
    ip = intersection_penalty(current_node, cyclist)
    tp = turn_penalty(current_edge, cyclist, predecessor)
    bpep = bike_path_end_penalty(current_edge, cyclist, on_bike_path, bike_path_ends)
    etp = edge_time_penalty(current_edge, cyclist)
    return (intersection=ip, turn=tp, bike_path_end=bpep, edge_time=etp)
end

"""
Returns the product of all multiplicative time penalties.
"""
function speed_penalty(current_edge, cyclist)
    slope_p = slope_penalty(current_edge, cyclist)
    car_p = car_penalty(current_edge, cyclist; check_bp=true)
    return slope_p * car_p
end

"""
Returns the felt length of an `edge` for a given `cyclist` with `speed_penalty` and aggregated `time_penalty`.
"""
function felt_edge_length(edge, speed_penalty, time_penalty, cyclist)
    return (weight(edge) / (speed_penalty * cyclist.speed)) + time_penalty
end