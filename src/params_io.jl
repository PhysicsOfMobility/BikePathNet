"""
Save an [`Experiment`](@ref) and a list of edge `bike_paths` which should have
bike paths on then to a `json` file at `file`.
"""
function save_comparison_params(file, experiment_config, bike_paths)
    # TODO: should we add input and output directories?
    data = @ntuple experiment_config bike_paths
    open(file, "w") do io
        print(io, to_pretty_json(data))
    end
end

"""
Load an [`Experiment`](@ref) and a list of edges `bike_paths` which should have
bike paths on them from a `json` file at `file`.
"""
function load_comparison_params(file)
    data = JSON.parsefile(file)
    experiment_config = Serde.deser(Experiment, data["experiment_config"])
    bike_paths = [(i...,) for i in data["bike_paths"]]
    return experiment_config, bike_paths
end