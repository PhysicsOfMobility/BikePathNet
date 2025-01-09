"""
Prepare (mutate) the graph `g` to be run with the settings in `algorithm_config`.

For example, if `algorithm_config.buildup==true`, all edges need to be
initialised with `edge.bike_path=true`, as the algorithm will work by
removing the bike path on the least used edges.
"""
function prepare_graph!(g::SpatialGraph, algorithm_config)
    prepare_graph!(g.graph, algorithm_config)
    return g
end
function prepare_graph!(g::SegmentGraph, algorithm_config)
    prepare_graph!(g.graph, algorithm_config)
    return g
end
function prepare_graph!(g::Graph, algorithm_config)
    for ((origin, destination), edge) in g.edges
        if (origin, destination) in algorithm_config.add_blocked_streets
            edge.blocked = true
        end

        # if we use existing infrastructure
        if algorithm_config.use_existing_infrastructure
            edge.bike_path = !algorithm_config.buildup || edge.ex_inf
        else
            edge.ex_inf = false
            edge.bike_path = !algorithm_config.buildup
        end

        # if we use blocked streets
        if algorithm_config.use_blocked_streets
            if edge.blocked
                # Stays a bike path if there is an exising one
                edge.bike_path = edge.ex_inf
                edge.edited = true
            end
        else
            edge.blocked = false
        end

        # if we use bike highways
        if algorithm_config.use_bike_highways
            if edge.ramp
                edge.edited = true
            end
        else
            edge.bike_highway = false
            edge.ramp = false
        end
    end

    # reset all nodes to not be bike highways if we do not use them
    if !algorithm_config.use_bike_highways
        for (node_id, node) in g.nodes
            if node.bike_highway
                g.nodes[node_id] = @set node.bike_highway = false
            end
        end
    end

    return g
end