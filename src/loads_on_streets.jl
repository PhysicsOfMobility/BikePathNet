"""
Type to communicate that the minimal loaded element is an edge, together with
its `source` and `destination`.
"""
struct MinLoadedEdge
    source::Int
    destination::Int
end

"""
Type to communicate that the minimal loaded element is a segment, together with
its `id`.
"""
struct MinLoadedSegment
    id::Int
end

"""
Turns [`e::MinLoadedEdge`](@ref MinLoadedEdge) into a tuple of the [`Edge`](@ref) in the graph `g` that
can be iterated over.
"""
function to_edge_iter(e::MinLoadedEdge, g)
    return (edges(g)[e.source, e.destination],)
end

"""
Turns [`s::MinLoadedSegment`](@ref MinLoadedSegment) into an iterator of the [`Edge`](@ref) in the graph `g`.
"""
function to_edge_iter(s::MinLoadedSegment, g)
    return (edges(g)[e] for e in g.segments[s.id].edges)
end

"Turns the [`e::MinLoadedEdge`](@ref MinLoadedEdge) into a tuple of `(source, destination)` for the gains dictionary."
to_gains_key(e::MinLoadedEdge) = (e.source, e.destination)
"Turns the [`s::MinLoadedSegment`](@ref MinLoadedSegment) into the `id` for the gains dictionary."
to_gains_key(s::MinLoadedSegment) = s.id

"Tests if the [`e::MinLoadedEdge`](@ref MinLoadedEdge) is non-existent. (i.e. `source` and `destination` are -1)"
is_non_existent(e::MinLoadedEdge) = e.source == -1 && e.destination == -1
"Tests if the [`s::MinLoadedSegment`](@ref MinLoadedSegment) is non-existent. (i.e. `id` is -1)"
is_non_existent(s::MinLoadedSegment) = s.id == -1

"""
Returns the minimal loaded element ([`MinLoadedEdge`](@ref)) of the graph [`g::Graph`](@ref Graph)
according to the [`minstate<:AbstractMinmodeState`](@ref AbstractMinmodeState)
and the [`algorithm_config`](@ref AlgorithmConfig).
"""
minimal_loaded_element(g::Graph, minstate, algorithm_config) = minimal_loaded_edge(g, minstate, algorithm_config)
"""
Returns the minimal loaded element ([`MinLoadedSegment`](@ref)) of the graph [`g::SegmentGraph`](@ref SegmentGraph)
according to the [`minstate<:AbstractMinmodeState`](@ref AbstractMinmodeState)
and the [`algorithm_config`](@ref AlgorithmConfig).
"""
minimal_loaded_element(g::SegmentGraph, minstate, algorithm_config) = minimal_loaded_segment(g, minstate, algorithm_config)

"""
Returns the [`MinLoadedSegment`](@ref) of the [`SegmentGraph`](@ref) `g` which should be edited next according to the `algorithm_config`.
"""
function minimal_loaded_segment(g::SegmentGraph, minstate, algorithm_config)
    # TODO: not super sure how this interacts with blocked streets and existing infrastructure...
    segment_loads = Dict{Int,Float64}()
    for (k, e) in edges(g)
        (e.edited || e.blocked) && continue  # TODO: after running prepare_graph! all blocked edges are already edited...
        if haskey(segment_loads, e.seg_id)
            segment_loads[e.seg_id] += _edge_load(e, g.segments[e.seg_id], minstate)
        else
            segment_loads[e.seg_id] = _edge_load(e, g.segments[e.seg_id], minstate)
        end
    end

    if length(segment_loads) == 0
        return MinLoadedSegment(-1), false
    else
        target_load = if algorithm_config.buildup
            maximum(values(segment_loads))
        else
            minimum(values(segment_loads))
        end
        segment_id = rand(filter(kv -> kv[2] == target_load, segment_loads))[1]
        return MinLoadedSegment(segment_id), target_load > 0.0
    end
end

"""
Returns the [`MinLoadedSegment`](@ref) of the [`SegmentGraph`](@ref) `g` which should be edited next according to the given fixed orden by the `minstate`.
"""
function minimal_loaded_segment(g::SegmentGraph, minstate::FixedOrderMinmodeState, _)
    segment_id = popfirst!(minstate.order)
    target_load = sum(_edge_load(g.graph.edges[e], g.segments[segment_id], minstate) for e in g.segments[segment_id].edges)

    return MinLoadedSegment(segment_id), target_load > 0.0
end

"""
Returns the [`MinLoadedEdge`](@ref) of the [`Graph`](@ref) `g` which should be edited next according to the `algorithm_config`.
"""
function minimal_loaded_edge(g::Graph, minstate, algorithm_config)
    best_edge = (-1, -1)
    best_load = algorithm_config.buildup ? -Inf : Inf
    edges_shuffled = shuffle(collect(keys(g.edges)))
    for k in edges_shuffled
        e = g.edges[k]
        (e.edited || e.blocked) && continue
        next_load = _edge_load(e, minstate)
        if algorithm_config.buildup
            if next_load > best_load
                best_edge = k
                best_load = next_load
            end
        else
            if next_load < best_load
                best_edge = k
                best_load = next_load
            end
        end
    end
    return MinLoadedEdge(best_edge...), best_load > 0.0
end

"""
Returns the [`MinLoadedEdge`](@ref) of the [`Graph`](@ref) `g` which should be edited next according to the given fixed orden by the `minstate`.
"""
function minimal_loaded_edge(g::Graph, minstate::FixedOrderMinmodeState, _)
    best_edge = popfirst!(minstate.order)
    best_load = _edge_load(g.edges[best_edge], minstate)

    return MinLoadedEdge(best_edge...), best_load > 0.0
end

"""
    _edge_load(e::Edge, minstate<:AbstractMinmodeState)
    _edge_load(e::Edge, seg::Segment, minstate<:AbstractMinmodeState)

Returns the load on the edge `e` according to the `minstate`.
If `seg` is provided, the return value is still the load on edge `e`.
Aggregation of the individual (edge-) loads is done in [`minimal_loaded_segment`](@ref).
"""
function _edge_load end

_edge_load(e, minstate::MinPureCyclistsState) = minstate.aggregated_loads[e.source, e.destination]
_edge_load(e, seg, minstate::MinPureCyclistsState) = _edge_load(e, minstate)
_edge_load(e, minstate::MinPenaltyWeightedCyclistsState) = minstate.aggregated_loads[e.source, e.destination]
_edge_load(e, seg, minstate::MinPenaltyWeightedCyclistsState) = _edge_load(e, minstate)
_edge_load(e, minstate::MinPenaltyCostWeightedCyclistsState) = minstate.aggregated_loads[e.source, e.destination] / e.cost
_edge_load(e, seg, minstate::MinPenaltyCostWeightedCyclistsState) = minstate.aggregated_loads[e.source, e.destination] / seg.cost
_edge_load(e, minstate::MinPenaltyLengthWeightedCyclistsState) = minstate.aggregated_loads[e.source, e.destination]
_edge_load(e, seg, minstate::MinPenaltyLengthWeightedCyclistsState) = _edge_load(e, minstate)
_edge_load(e, minstate::MinPenaltyLengthCostWeightedCyclistsState) = minstate.aggregated_loads[e.source, e.destination] / e.length
_edge_load(e, seg, minstate::MinPenaltyLengthCostWeightedCyclistsState) = minstate.aggregated_loads[e.source, e.destination] / seg.length
_edge_load(e, minstate::TravelTimeBenefitsState) = minstate.aggregated_loads[e.source, e.destination]
_edge_load(e, seg, minstate::TravelTimeBenefitsState) = _edge_load(e, minstate)
_edge_load(e, minstate::TravelTimeBenefitsCostState) = minstate.aggregated_loads[e.source, e.destination] / e.cost
_edge_load(e, seg, minstate::TravelTimeBenefitsCostState) = minstate.aggregated_loads[e.source, e.destination] / seg.cost
_edge_load(e, minstate::TravelTimeBenefitsHealthBenefitsState) = minstate.aggregated_loads[e.source, e.destination]
_edge_load(e, seg, minstate::TravelTimeBenefitsHealthBenefitsState) = _edge_load(e, minstate)
_edge_load(e, minstate::TravelTimeBenefitsHealthBenefitsCostState) = minstate.aggregated_loads[e.source, e.destination] / e.cost
_edge_load(e, seg, minstate::TravelTimeBenefitsHealthBenefitsCostState) = minstate.aggregated_loads[e.source, e.destination] / seg.cost

_edge_load(e, minstate::MinTotalDetourState) = minstate.aggregated_detours[e.source, e.destination]
_edge_load(e, seg, minstate::MinTotalDetourState) = minstate.aggregated_detours[e.source, e.destination]

_edge_load(e, minstate::FixedOrderMinmodeState) = minstate.aggregated_loads[e.source, e.destination]
_edge_load(e, seg, minstate::FixedOrderMinmodeState) = _edge_load(e, minstate)

"""
Edits the [`Edge`](@ref) in `e` by flipping the `bike_path` and setting `edited` to `true`.
"""
function edit_edge!(e)
    e.edited = true
    e.bike_path = !e.bike_path
end

"""
Un-Edits the [`Edge`](@ref) in `e` by flipping the `bike_path` and setting `edited` to `false`.
Used to reset an edit when calculating the [`MinTotalDetour`](@ref) minmode.
"""
function edit_edge_back!(e)
    e.edited = false
    e.bike_path = !e.bike_path
end