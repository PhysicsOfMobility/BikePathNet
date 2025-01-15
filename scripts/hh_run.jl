using DrWatson
@quickactivate :BikePathNet
using Setfield

city = "Hamburg"
load = "hh"
save = "hh"

cyclist_types = 1

params = Dict()
params["speeds"] = [1]
params["car_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:primary, [1/7]),
    (:secondary, [1/2.4]),
    (:tertiary, [1/1.4]),
    (:residential, [1/1.1]),
    (:bike_path, [1]),
    ])
params["slope_penalty"] = Dict([
    (0.06, [1/3.2]),
    (0.04, [1/2.2]),
    (0.02, [1/1.4]),
    (0.0, [1/1.0])
    ])
params["intersection_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, [25]),
    (:medium, [15]),
    (:small, [5])
    ])
params["turn_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, [20]),
    (:medium, [10]),
    (:small, [0])
    ])
params["edge_time_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, [0]),
    (:medium, [0]),
    (:small, [0])
    ])
params["bp_end_penalty"] = Dict{Symbol,Dict{Int,Vector{Float64}}}([
    (:primary, Dict([(1, [10]), (2, [15]), (3, [25])])),
    (:secondary, Dict([(1, [5]), (2, [10]), (3, [15])])),
    (:tertiary, Dict([(1, [0]), (2, [5]), (3, [7.5])])),
    (:residential, Dict([(1, [0]), (2, [1]), (3, [1])])),
    (:bike_path, Dict([(1, [0])])),
    ])

cyclists = BikePathNet.Cyclist[]
for i in 1:cyclist_types
    push!(cyclists, BikePathNet.Cyclist(
        id=i,
        speed=params["speeds"][i],
        car_penalty=Dict(k => v[i] for (k, v) in params["car_penalty"]),
        slope_penalty=Dict(k => v[i] for (k, v) in params["slope_penalty"]),
        intersection_penalty=Dict(k => v[i] for (k, v) in params["intersection_penalty"]),
        edge_time_penalty=Dict(k => v[i] for (k, v) in params["edge_time_penalty"]),
        turn_penalty=Dict(k => v[i] for (k, v) in params["turn_penalty"]),
        bikepath_end_penalty=Dict(k => Dict(k_j => v_j[i] for (k_j, v_j) in v) for (k, v) in params["bp_end_penalty"]),
        value_hb=1.0
    ))
end

config = BikePathNet.AlgorithmConfig(
            buildup=false,
            use_existing_infrastructure=false,
            use_blocked_streets=false,
            add_blocked_streets=[(-1, -1)],
            use_bike_highways=false,
            save_edge_loads=false,
            undirected=false,
            minmode=BikePathNet.MinPureCyclists(),
            beta=0.0,
        )
experiment = BikePathNet.Experiment(
        save=save,
        city_name=city,
        algorithm_config=config,
        cyclists=cyclists,
        graph_type=BikePathNet.Graph,
        trip_type=BikePathNet.Trip,
        graph_file=["input", load, "$(load).graphml"],
        trips_file=["input", load, "$(load)_demand.json"],
        output_file=["output", save, "$(save)_data.json"],
        log_file=["logs", "$(save).log"],
    )

modes = Dict(
    "010" => (BikePathNet.MinPenaltyWeightedCyclists(), false),
    "011" => (BikePathNet.MinPenaltyWeightedCyclists(), true),
    )

for (mode_str, mode) in modes
    alg_config = BikePathNet.AlgorithmConfig(
            buildup=config.buildup,
            use_existing_infrastructure=mode[2],
            use_blocked_streets=false,
            add_blocked_streets=[(-1, -1)],
            use_bike_highways=config.use_bike_highways,
            save_edge_loads=false,
            undirected=config.undirected,
            minmode=mode[1],
            beta=config.beta,
        )

    alg_experiment = BikePathNet.Experiment(
        save=experiment.save,
        city_name=experiment.city_name,
        algorithm_config=alg_config,
        cyclists=experiment.cyclists,
        graph_type=experiment.graph_type,
        trip_type=experiment.trip_type,
        graph_file=["input", load, "$(load).graphml"],
        trips_file=["input", load, "$(load)_demand.json"],
        output_file=["output", save, "$(save)_data_mode_$(mode_str).json"],
        log_file=["logs", "$(save)_$(mode_str).log"],
    )
    alg_experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode_str).json")
    BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
    BikePathNet.run_simulation(alg_experiment_file)

    mode[2] ? cs_save = "cs_ex_inf" : cs_save = "cs"
    cs_config = @set alg_config.save_edge_loads = false
    cs_config = @set cs_config.minmode = BikePathNet.MinPureCyclists()
    cs_experiment = @set alg_experiment.algorithm_config = cs_config
    cs_experiment = @set cs_experiment.output_file = ["output", save, "$(save)_data_comparison_state_$(cs_save).json"]
    cs_experiment_file = datadir("input", save, "$(save)_comparison_state_params_$(cs_save).json");
    cs_bike_paths = BikePathNet.get_ps_bp_from_algorithm_results(datadir(alg_experiment.output_file...))
    BikePathNet.save_comparison_params(cs_experiment_file, cs_experiment, cs_bike_paths);
    BikePathNet.calculate_comparison_state(cs_experiment_file);

    edge_load_experiment = @set cs_experiment.output_file = ["output", save, "$(save)_data_algo_comp_edge_load_$(mode_str).json"]
    edge_load_experiment_file = datadir("input", save, "$(save)_algo_comp_edge_load_params_$(mode_str).json");
    edge_load_bike_paths = BikePathNet.get_comparison_bp_from_algorithm_results(datadir(alg_experiment.output_file...), datadir(cs_experiment.output_file...));
    BikePathNet.save_comparison_params(edge_load_experiment_file, edge_load_experiment, edge_load_bike_paths);
    BikePathNet.calculate_comparison_state(edge_load_experiment_file);
end

base_experiment = @set experiment.output_file = ["output", save, "$(save)_data_comparison_state_base.json"]
base_experiment_file = datadir("input", save, "$(save)_comparison_state_params_base.json");
BikePathNet.save_comparison_params(base_experiment_file, base_experiment, Tuple{Int,Int}[]);
BikePathNet.calculate_base_comparison_state(base_experiment_file);

opt_experiment = @set experiment.output_file = ["output", save, "$(save)_data_comparison_state_opt.json"]
opt_experiment_file = datadir("input", save, "$(save)_comparison_state_params_opt.json");
BikePathNet.save_comparison_params(opt_experiment_file, opt_experiment, Tuple{Int,Int}[]);
BikePathNet.calculate_optimal_comparison_state(opt_experiment_file);

exinf_config = @set config.use_existing_infrastructure = true
exinf_experiment = @set experiment.algorithm_config = exinf_config
exinf_experiment = @set exinf_experiment.output_file = ["output", save, "$(save)_data_comparison_state_ex_inf.json"]
exinf_experiment_file = datadir("input", save, "$(save)_comparison_state_params_ex_inf.json");
BikePathNet.save_comparison_params(exinf_experiment_file, exinf_experiment, []);
BikePathNet.calculate_comparison_state(exinf_experiment_file);