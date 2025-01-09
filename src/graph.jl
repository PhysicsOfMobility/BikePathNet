"""
Node of a [`Graph`](@ref), [`SegmentGraph`](@ref) or [`SpatialGraph`](@ref)
which represents an intersection in the street network.
"""
struct Node
    "unique id of the node"
    id::Int
    "size of the intersection"
    size::Symbol
    "`true` if the node is part of a bike highway, `false` otherwise"
    bike_highway::Bool
end

"""
Edge of a [`Graph`](@ref), [`SegmentGraph`](@ref) or [`SpatialGraph`](@ref)
which represents a directed street in the street network.
"""
@kwdef mutable struct Edge
    "source id of the street"
    source::Int
    "destination id of the street"
    destination::Int
    "physical length of the street in meters"
    length::Float64
    "type of the street (e.g. `:residential`, `:primary`)"
    street_type::Symbol
    "slope of the street in percent (rise over run)"
    slope::Float64
    "size of penalty when turning from this edge onto another edge, indexed by destination node id of the edge we turn on"
    turn_badness::Dict{Int,Symbol}
    "size of penalty for travelling along this edge"
    time_penalty::Symbol
    "`true` if there is existing bike infrastructure on this edge, `false` otherwise"
    ex_inf::Bool
    "`true` if no infrastructure can be built on this edge, `false` otherwise"
    blocked::Bool
    "`true` if there exists a bike path along this edge, `false` otherwise"
    bike_path::Bool
    "`true` if the edge is part of a bike highway, `false` otherwise"
    bike_highway::Bool
    "`true` if the edge is a ramp leading on or of a bike highway, `false` otherwise"
    ramp::Bool
    "cost of build a bike path along this edge"
    cost::Float64
    "id of the segment the edge belongs to (only used with the [`SegmentGraph`](@ref))"
    seg_id::Int
    "`true` if this edge has already been considered for a bike path addition/removal, `false` otherwise"
    edited::Bool
end

"""
Connected run of edges which are supposed to be built together.
"""
struct Segment
    "unique id of the segment"
    id::Int
    "cost of building this segment"
    cost::Float64
    "physical length of the segment in meters"
    length::Float64
    "edges which are part of the segment"
    edges::Vector{Tuple{Int,Int}}
end

function Base.show(io::IO, ::MIME"text/plain", thing::T) where {T<:Union{Node,Edge,Segment}}
    println(io, T)
    for field in fieldnames(T)
        println(io, "  ", field, ": ", getfield(thing, field))
    end
end

##### Simple graph #######
"""
Graph representing a simple street network.
"""
struct Graph <: AbstractGraph
    "intersections of network"
    nodes::Dict{Int,Node}
    "outbound (directed) adjacency between nodes"
    adjacency_lists::Dict{Int,Set{Int}}
    "streets (edges) between the intersections (nodes)"
    edges::Dict{Tuple{Int,Int},Edge}
end

"Empty constructor for a [`Graph`](@ref)."
Graph() = Graph(Dict(), Dict(), Dict())

nv(g::Graph) = length(g.nodes)
ne(g::Graph) = length(g.edges)
nodes(g::Graph) = g.nodes
edges(g::Graph) = g.edges

neighbours(g::Graph, node) = g.adjacency_lists[node]

##### Segment graph #######
"""
Graph representing a simple street network with segments which are beeing built/removed together.
"""
struct SegmentGraph <: AbstractGraph
    "holding the underlying graph structure"
    graph::Graph
    "segments of the network indexed by their id"
    segments::Dict{Int,Segment}
end
"Empty constructor for a [`SegmentGraph`](@ref)."
SegmentGraph() = SegmentGraph(Graph(), Dict())

nv(g::SegmentGraph) = nv(g.graph)
ne(g::SegmentGraph) = ne(g.graph)
nodes(g::SegmentGraph) = nodes(g.graph)
edges(g::SegmentGraph) = edges(g.graph)

neighbours(g::SegmentGraph, node) = neighbours(g.graph, node)

##### Spatial graph #######
"""
Graph representing a street network with spatial information about the nodes and edges,
useful for visualization.
"""
struct SpatialGraph{G<:AbstractGraph}
    "underlying graph structure"
    graph::G
    "coordinates of nodes indexed by node id"
    node_geometry::Dict{Int,Any}
    "coordinates of edge-geometry indexed by source and destination node id"
    edge_geometry::Dict{Tuple{Int,Int},Any}
end
"""
    SpatialGraph{G<:BikePathNet.AbstractGraph}()
Empty constructor for a [`SpatialGraph`](@ref).
"""
SpatialGraph{T}() where {T<:AbstractGraph} = SpatialGraph(T(), Dict{Int,Any}(), Dict{Tuple{Int,Int},Any}())

nv(g::SpatialGraph) = nv(g.graph)
ne(g::SpatialGraph) = ne(g.graph)

function Base.show(io::IO, ::MIME"text/plain", g::T) where {T<:Union{Graph,SegmentGraph,SpatialGraph}}
    println(io, T, " with ", nv(g), " nodes and ", ne(g), " edges.")
end