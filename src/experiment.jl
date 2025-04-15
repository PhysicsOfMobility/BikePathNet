"""
Full description of an experiment.
"""
@kwdef struct Experiment
    "name of experiment for logging (TODO??)"
    save::String
    "name of the city"
    city_name::String
    "configuration for the algorithm"
    algorithm_config::AlgorithmConfig
    "cyclists to be used in the experiment"
    cyclists::Vector{Cyclist}
    "type of the trips"
    trip_type::DataType
    "type of the graph"
    graph_type::DataType
    "path to the graph file, relative to `datadir(...)`."
    graph_file::Vector{String}
    "path to the trips file, relative to `datadir(...)`."
    trips_file::Vector{String}
    "path to the output file, relative to `datadir(...)`."
    output_file::Vector{String}
    "path to the log file, relative to `datadir(...)`. Empty vector if no log file is used."
    log_file::Vector{String}
end

function Base.show(io::IO, ::MIME"text/plain", experiment::T) where {T<:Experiment}
    println(io, T)
    for field in fieldnames(T)
        println(io, "  ", field, ": ", getfield(experiment, field))
    end
end

#### Serde overloads ####
function Serde.SerJson.ser_type(::Type{<:Experiment}, v::T) where {T<:DataType}
    return split(string(v), '.')[end]
end

function Serde.deser(::Type{<:Experiment}, ::Type{DataType}, string)
    if string == "Graph"
        return Graph
    elseif string == "SegmentGraph"
        return SegmentGraph
    elseif string == "Trip"
        return Trip
    elseif string == "VariableTrip"
        return VariableTrip
    else
        throw(ArgumentError("Unknown trip or graph type: $string"))
    end
end

#### saving and loading ####
"""
Write an [`Experiment`](@ref) to the `json` file at `file`.
"""
function save_experiment(file, experiment::Experiment)
    mkpath(dirname(file))
    open(file, "w") do io
        print(io, to_pretty_json(experiment))
    end
end

"""
Load the `json` file at `file` into an [`Experiment`](@ref).
"""
function load_experiment(file)
    data = JSON.parsefile(file)
    return Serde.deser(Experiment, data)
end


"Get the location of the graph file from an [`Experiment`](@ref)."
graph_file(experiment) = datadir(experiment.graph_file...)
"Get the location of the trips file from an [`Experiment`](@ref)."
trips_file(experiment) = datadir(experiment.trips_file...)
"Get the location of the output file from an [`Experiment`](@ref)."
output_file(experiment) = datadir(experiment.output_file...)
"Get the location of the log file from an [`Experiment`](@ref). Returns `nothing` if no log file is used."
log_file(experiment) = length(experiment.log_file) > 0 ? datadir(experiment.log_file...) : nothing

"Load the graph from an [`Experiment`](@ref), using the specified graph type and prepare it."
function load_graph(experiment::Experiment)
    g = load_graph(graph_file(experiment), experiment.graph_type)
    return prepare_graph!(g, experiment.algorithm_config)
end

"Load the trips from an [`Experiment`](@ref), using the specified trip type."
load_trips(experiment::Experiment) = load_trips(trips_file(experiment), experiment.cyclists, experiment.trip_type)