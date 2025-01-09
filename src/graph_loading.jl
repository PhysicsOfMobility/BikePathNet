"""
Loads the `graphml` file at `path` into a `graphtype`.
"""
function load_graph(path, graphtype)
    g = open(path, "r") do io
        reader = EzXML.StreamReader(io)
        keys = Dict{String,Dict{String,String}}()
        for typ in reader
            if typ == EzXML.READER_ELEMENT
                elname = EzXML.nodename(reader)
                # extract all relevant keys
                if elname == "key"
                    attrs = EzXML.nodeattributes(reader)
                    keys[attrs["id"]] = attrs
                elseif elname == "graph"
                    # parse the first graph we come across
                    parsed_graph = parse_graph!(reader, keys, graphtype)
                    close(reader)
                    return parsed_graph
                end
            end
        end
    end
    return g
end


"""
Parses the `reader` into a graph of type `graphtype`.
`keys::Dict{String,Dict{String,String}}` map the `key` attributes of the nodes
and edges to the attributes of the `key` nodes.

Assumes current node of the `reader` to be the `graph` node, advances the reader.
"""
function parse_graph!(reader, keys, graphtype)
    g = graphtype()

    graph_data = Dict{String,String}()

    nodes = []
    edges = []

    current_depth = EzXML.nodedepth(reader)
    for typ in reader
        if typ == EzXML.READER_ELEMENT
            elname = EzXML.nodename(reader)
            if elname == "node"
                node_data = parse_node!(reader, keys)
                unsafe_add_node!(g, node_data...)
                push!(nodes, node_data[1])
            elseif elname == "edge"
                edge_data = parse_edge!(reader, keys)
                unsafe_add_edge!(g, edge_data...)
                push!(edges, edge_data[1])
            elseif elname == "data"
                data_key = reader["key"]
                keyname = keys[data_key]["attr.name"]
                graph_data[keyname] = EzXML.nodecontent(reader)
            else
                @warn "There is an unexpected $(elname) in your Graph."
            end
        end
        if current_depth >= EzXML.nodedepth(reader)
            break
        end
    end
    fix_edge_geometry_order!(g)
    fix_bike_highway_flags!(g)

    return g
end

"""
Flips the direction of the edge-geometries in `g` such, that the first point in the line is the same
(closest to) as the source node of each edge. Does nothing if `g` does not contain edge geometries.
"""
fix_edge_geometry_order!(g::AbstractGraph) = g
function fix_edge_geometry_order!(g::SpatialGraph)
    for (k, e) in g.edge_geometry
        # kind of involved, but just to be sure...
        n1 = g.node_geometry[k[1]]
        n2 = g.node_geometry[k[2]]
        e1 = e[1]
        e2 = e[end]
        distmat = _distance_node_edge.([n1 n2], [e1, e2])
        if distmat[1, 1] == distmat[2, 2] == 0 && distmat[1, 2] == distmat[2, 1] != 0.0
        elseif distmat[1, 1] == distmat[2, 2] != 0 && distmat[1, 2] == distmat[2, 1] == 0.0
            reverse!(e)  # somehow the edges are loaded from destination to start?
        else
            @warn "Something went wrong when trying to establish wether to reverse the edgegeometry on edge $k (distmat=$distmat)"
        end
    end
    return g
end


_distance_node_edge(n, e) = sqrt((n.lon - e[1])^2 + (n.lat - e[2])^2)

"""
Correctly sets the `bike_highway` flag of each node according to the connected edges.
(If the edge is a bike highway and not a ramp, the flag is set on the source and destination.)
"""
fix_bike_highway_flags!(g::SpatialGraph) = fix_bike_highway_flags!(g.graph)
fix_bike_highway_flags!(g::SegmentGraph) = fix_bike_highway_flags!(g.graph)
function fix_bike_highway_flags!(g::Graph)
    for (k, e) in g.edges
        if e.bike_highway && !e.ramp
            source_node = g.nodes[e.source]
            destination_node = g.nodes[e.destination]
            g.nodes[e.source] = @set source_node.bike_highway = true
            g.nodes[e.destination] = @set destination_node.bike_highway = true
        end
    end
end


"""
parsers the `reader` into a [`Node`](@ref) and some spatial information.
Assumes current node of the `reader` to be the `node` node.
"""
function parse_node!(reader, keys)
    id = parse(Int, reader["id"])

    node_dict = parse_data_dict!(reader, keys)

    intersection_size = Symbol(node_dict["intersection_size"])
    node = Node(
        id,
        intersection_size,
        false  # This is set in fix_bike_highway_flags! after we know all adjacent edges
    )
    geom = (
        lon=parse(Float64, node_dict["x"]),
        lat=parse(Float64, node_dict["y"])
    )

    return (node, geom)
end


"""
parsers the `reader` into an [`Edge`](@ref) and some spatial information.
Assumes current node of the `reader` to be the `edge` node.
"""
function parse_edge!(reader, keys)
    source = parse(Int, reader["source"])
    destination = parse(Int, reader["target"])

    edge_dict = parse_data_dict!(reader, keys)

    # read values needed multiple times
    street_length = parse(Float64, edge_dict["length"])
    street_type = Symbol(edge_dict["highway"])
    slope = parse(Float64, edge_dict["grade"])
    parsed_turn_badness = parse_turn_badness(edge_dict["turn_penalty"])

    # create with default values where needed.
    edge = Edge(
        source=source,
        destination=destination,
        length=street_length,
        street_type=street_type,
        slope=slope,
        turn_badness=parsed_turn_badness,
        time_penalty=Symbol(get(edge_dict, "time_penalty", "small")),
        ex_inf=parsebool(edge_dict["ex_inf"]),
        blocked=parsebool(get(edge_dict, "blocked", "false")),
        bike_path=true,  # This is set to the correct value when preparing for algorithm with config
        bike_highway=parsebool(get(edge_dict, "bike_highway", "false")),
        ramp=parsebool(get(edge_dict, "ramp", "false")),
        cost=parse(Float64, edge_dict["cost"]),
        seg_id=parse(Int, get(edge_dict, "seg_id", "0")),
        edited=false,  # This is set to the correct value when preparing for algoritm with config
    )

    geom = parse_linestring(edge_dict["geometry"])
    return (edge, geom)
end


"""
Parses the `data` nodes contained in the current `reader` into a `Dict{String, String}`,
using the `keys` which map the `data.key` attribute to human readable keys.
"""
function parse_data_dict!(reader, keys)
    data_dict = Dict{String,String}()

    current_depth = EzXML.nodedepth(reader)
    for typ in reader
        if typ == EzXML.READER_ELEMENT
            data_key = reader["key"]
            keyname = keys[data_key]["attr.name"]
            data_dict[keyname] = EzXML.nodecontent(reader)
        end
        if current_depth >= EzXML.nodedepth(reader)
            break
        end
    end
    return data_dict
end

"""
Adds the node `node` with geometry `geom` to the graph `g`, possibly overwriting preexisting data.
"""
unsafe_add_node!(g::SegmentGraph, node::Node, geom) = unsafe_add_node!(g.graph, node, geom)
function unsafe_add_node!(g::Graph, node::Node, _)
    g.nodes[node.id] = node
end

function unsafe_add_node!(g::SpatialGraph, node::Node, geom)
    unsafe_add_node!(g.graph, node, geom)
    g.node_geometry[node.id] = geom
end

"""
Adds the edge `edge` with geometry `geom` to the graph `g`, not checking for data-integrity.
"""
function unsafe_add_edge!(g::SegmentGraph, edge::Edge, geom)
    unsafe_add_edge!(g.graph, edge, geom)
    if haskey(g.segments, edge.seg_id)
        segment = g.segments[edge.seg_id]
        push!(segment.edges, (edge.source, edge.destination))
        g.segments[edge.seg_id] = @set segment.length += edge.length
    else
        g.segments[edge.seg_id] = Segment(edge.seg_id, edge.cost, edge.length, [(edge.source, edge.destination)])
    end
end

function unsafe_add_edge!(g::Graph, edge::Edge, _)
    s = edge.source
    d = edge.destination

    g.edges[(s, d)] = edge

    if !(haskey(g.adjacency_lists, s))
        g.adjacency_lists[s] = Set()
    end
    push!(g.adjacency_lists[s], d)
end

function unsafe_add_edge!(g::SpatialGraph, edge::Edge, geom)
    unsafe_add_edge!(g.graph, edge, geom)
    g.edge_geometry[(edge.source, edge.destination)] = geom
end


function parse_turn_badness(tp_string)
    valid_json = replace(replace(tp_string, r"(\d+)" => s"\"\1\""), r"'" => s"\"")
    tp_dict = JSON.parse(valid_json)
    return Dict(parse(Int, key) => Symbol(value) for (key, value) in tp_dict)
end

parsebool(string) = parse(Bool, lowercase(string))

function parse_linestring(line)
    @assert startswith(line, "LINESTRING")
    rx = r"\d+\.?\d*"
    values = [parse(Float64, m.match) for m in eachmatch(rx, line)]
    return Iterators.partition(values, 2) |> collect
end