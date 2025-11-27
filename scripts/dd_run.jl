using DrWatson
@quickactivate :BikePathNet
using Setfield
using JSON
using Serde

city = "Dresden"
base_load = "dd"
base_save = "dd"

random_realisations = 10
noisy_data_sets = 0

cyclist_types = 1

params = Dict()
params["speeds"] = [17.8/3.6]
params["car_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:primary, [1/7]),
    (:secondary, [1/2.4]),
    (:tertiary, [1/1.4]),
    (:residential, [1/1.1]),
    (:bike_path, [1]),
    (:superhighway, [1]),
    ])
params["slope_penalty"] = Dict([
    (0.06, [1/3.2]),
    (0.04, [1/2.2]),
    (0.02, [1/1.4]),
    (0.0, [1])
    ])
params["intersection_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, [30]),
    (:medium, [15]),
    (:small, [0])
    ])
params["turn_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, [15]),
    (:medium, [7.5]),
    (:small, [0])
    ])
params["edge_time_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, [0]),
    (:medium, [0]),
    (:small, [0])
    ])
params["bp_end_penalty"] = Dict{Symbol,Dict{Int,Vector{Float64}}}([
    (:primary, Dict([(1, [0]), (2, [0]), (3, [0])])),
    (:secondary, Dict([(1, [0]), (2, [0]), (3, [0])])),
    (:tertiary, Dict([(1, [0]), (2, [0]), (3, [0])])),
    (:residential, Dict([(1, [0]), (2, [0]), (3, [0])])),
    (:bike_path, Dict([(1, [0])])),
    (:superhighway, Dict([(1, [0])])),
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
            minmode=BikePathNet.PureCyclists(),
            beta=0.0,
        )
experiment = BikePathNet.Experiment(
        save=base_save,
        city_name=city,
        algorithm_config=config,
        cyclists=cyclists,
        graph_type=BikePathNet.Graph,
        trip_type=BikePathNet.Trip,
        graph_file=["input", base_load, "$(base_load).graphml"],
        trips_file=["input", base_load, "$(base_load)_demand.json"],
        output_file=["output", base_save, "$(base_save)_data.json"],
        log_file=["logs", "$(base_save).log"],
    )

modes = Dict(
    "000" => (BikePathNet.CyclistsLengthNormed(), false),
    "001" => (BikePathNet.CyclistsLengthNormed(), true),
    "010" => (BikePathNet.NormalizedTravelTimePenalty(), false),
    "011" => (BikePathNet.NormalizedTravelTimePenalty(), true),
    "020" => (BikePathNet.TotalTravelTimeEstimation(), false),
    "021" => (BikePathNet.TotalTravelTimeEstimation(), true),
    "030" => (BikePathNet.TotalTravelTimeEstimationPerLength(), false),
    "031" => (BikePathNet.TotalTravelTimeEstimationPerLength(), true),
    "040" => (BikePathNet.TotalTravelTimeEstimationCapped(), false),
    "041" => (BikePathNet.TotalTravelTimeEstimationCapped(), true),
    "050" => (BikePathNet.TotalTravelTimeEstimationCappedPerLength(), false),
    "051" => (BikePathNet.TotalTravelTimeEstimationCappedPerLength(), true),
    )

for (mode_str, mode) in modes
    save = base_save
    load = base_load

    GC.gc()
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
        output_file=["output", save, "$(save)_data_$(mode_str).json"],
        log_file=["logs", "$(save)_$(mode_str).log"],
    )
    alg_experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode_str).json")
    BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
    BikePathNet.run_simulation(alg_experiment_file)

    res = load_core_algorithm_result(datadir("output", save, "$(save)_data_$(mode_str).json"))

    bp_order_file = datadir("output", save, "$(save)_order_$(mode_str).json")
    open(bp_order_file, "w") do io
        write(io, to_json(res.edited_edges))
    end
    utility_file = datadir("output", save, "$(save)_utility_$(mode_str).json")
    open(utility_file, "w") do io
        write(io, to_json(res.total_utility))
    end
    bpp_file = datadir("output", save, "$(save)_bpp_$(mode_str).json")
    open(bpp_file, "w") do io
        write(io, to_json(res.bike_path_percentage))
    end
end

random_modes = Dict(
    "080" => ("Random", false),
    "081" => ("Random", true),
    )
for n in 1:random_realisations
    for (mode_str, mode) in random_modes
        save = base_save
        load = base_load

        fixed_order = JSON.parsefile(datadir("input", save, "$(save)_random_order_$(lpad(n, 2, "0")).json"))

        GC.gc()
        alg_config = BikePathNet.AlgorithmConfig(
                buildup=config.buildup,
                use_existing_infrastructure=mode[2],
                use_blocked_streets=false,
                add_blocked_streets=[(-1, -1)],
                use_bike_highways=config.use_bike_highways,
                save_edge_loads=false,
                undirected=config.undirected,
                minmode=BikePathNet.FixedOrderMinmode(fixed_order),
                beta=config.beta,
            )

        alg_experiment = BikePathNet.Experiment(
            save="$(save)_random_$(lpad(n, 2, "0"))",
            city_name=experiment.city_name,
            algorithm_config=alg_config,
            cyclists=experiment.cyclists,
            graph_type=experiment.graph_type,
            trip_type=experiment.trip_type,
            graph_file=["input", load, "$(load).graphml"],
            trips_file=["input", load, "$(load)_demand.json"],
            output_file=["output", save, "$(save)_data_$(mode_str)_$(lpad(n, 2, "0")).json"],
            log_file=["logs", "$(save)_$(mode_str).log"],
        )
        alg_experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode_str)_$(lpad(n, 2, "0")).json")
        BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
        BikePathNet.run_simulation(alg_experiment_file)

        res = load_core_algorithm_result(datadir("output", save, "$(save)_data_$(mode_str)_$(lpad(n, 2, "0")).json"))

        bp_order_file = datadir("output", save, "$(save)_order_$(mode_str)_$(lpad(n, 2, "0")).json")
        open(bp_order_file, "w") do io
            write(io, to_json(res.edited_edges))
        end
        utility_file = datadir("output", save, "$(save)_utility_$(mode_str)_$(lpad(n, 2, "0")).json")
        open(utility_file, "w") do io
            write(io, to_json(res.total_utility))
        end
        bpp_file = datadir("output", save, "$(save)_bpp_$(mode_str)_$(lpad(n, 2, "0")).json")
        open(bpp_file, "w") do io
            write(io, to_json(res.bike_path_percentage))
        end
    end
end

noisy_saves =  vcat(["_noisy_demand_"*$(lpad(i, 2, "0")) for i in 1:noisy_data_sets], ["_noisy_car_penalty_"*$(lpad(i, 2, "0")) for i in 1:noisy_data_sets], ["_noisy_slope_penalty_"*$(lpad(i, 2, "0")) for i in 1:noisy_data_sets], ["_noisy_intersection_penalty_"*$(lpad(i, 2, "0")) for i in 1:noisy_data_sets], ["_noisy_turn_penalty_"*$(lpad(i, 2, "0")) for i in 1:noisy_data_sets])
eval_saves = noisy_saves

modes_noisy = Dict(
    "000" => (BikePathNet.CyclistsLengthNormed(), false),
    "001" => (BikePathNet.CyclistsLengthNormed(), true),
    "010" => (BikePathNet.NormalizedTravelTimePenalty(), false),
    "011" => (BikePathNet.NormalizedTravelTimePenalty(), true),
    "020" => (BikePathNet.TotalTravelTimeEstimation(), false),
    "021" => (BikePathNet.TotalTravelTimeEstimation(), true),
    "030" => (BikePathNet.TotalTravelTimeEstimationPerLength(), false),
    "031" => (BikePathNet.TotalTravelTimeEstimationPerLength(), true),
    "040" => (BikePathNet.TotalTravelTimeEstimationCapped(), false),
    "041" => (BikePathNet.TotalTravelTimeEstimationCapped(), true),
    "050" => (BikePathNet.TotalTravelTimeEstimationCappedPerLength(), false),
    "051" => (BikePathNet.TotalTravelTimeEstimationCappedPerLength(), true),
    )

for noisy_save in noisy_saves
    load = base_load*noisy_save
    save = base_save*noisy_save

    params_noisy = Dict()

    penalties = JSON.parsefile(datadir("input", save, "$(save)_penalties.json"))
    
    params_noisy["speeds"] = [1]
    params_noisy["car_penalty"] = get(penalties, "car_penalty", params["car_penalty"])
    params_noisy["slope_penalty"] = get(penalties, "slope_penalty", params["slope_penalty"])
    params_noisy["intersection_penalty"] = get(penalties, "intersection_penalty", params["intersection_penalty"])
    params_noisy["edge_time_penalty"] = get(penalties, "edge_time_penalty", params["edge_time_penalty"])
    params_noisy["turn_penalty"] = get(penalties, "turn_penalty", params["turn_penalty"])
    params_noisy["bp_end_penalty"] = Dict([
        (:primary, Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        (:secondary, Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        (:tertiary, Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        (:residential, Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        (:bike_path, Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        ])

    cyclists_noisy = BikePathNet.Cyclist[]
    for i in 1:cyclist_types
        push!(cyclists_noisy, BikePathNet.Cyclist(
            id=i,
            speed=params_noisy["speeds"][i],
            car_penalty=Dict(Symbol(k) => v[i] for (k, v) in params_noisy["car_penalty"]),
            slope_penalty=Dict(parse(Float64, k) => v[i] for (k, v) in params_noisy["slope_penalty"]),
            intersection_penalty=Dict(Symbol(k) => v[i] for (k, v) in params_noisy["intersection_penalty"]),
            edge_time_penalty=Dict(Symbol(k) => v[i] for (k, v) in params_noisy["edge_time_penalty"]),
            turn_penalty=Dict(Symbol(k) => v[i] for (k, v) in params_noisy["turn_penalty"]),
            bikepath_end_penalty=Dict(k => Dict(k_j => v_j[i] for (k_j, v_j) in v) for (k, v) in params_noisy["bp_end_penalty"]),
            value_hb=1.0
        ))
    end

    config_noisy = BikePathNet.AlgorithmConfig(
            buildup=false,
            use_existing_infrastructure=false,
            use_blocked_streets=false,
            add_blocked_streets=[(-1, -1)],
            use_bike_highways=false,
            save_edge_loads=false,
            undirected=false,
            minmode=BikePathNet.PureCyclists(),
            beta=0.0,
        )

    experiment_noisy = BikePathNet.Experiment(
        save=save,
        city_name=experiment.city_name,
        algorithm_config=config_noisy,
        cyclists=cyclists_noisy,
        graph_type=experiment.graph_type,
        trip_type=experiment.trip_type,
        graph_file=["input", load, "$(load).graphml"],
        trips_file=["input", load, "$(load)_demand.json"],
        output_file=["output", save, "$(save)_data.json"],
        log_file=["logs", "$(save).log"],
        )

    for (mode_str, mode) in modes_noisy
        GC.gc()
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
            save=experiment_noisy.save,
            city_name=experiment_noisy.city_name,
            algorithm_config=alg_config,
            cyclists=experiment_noisy.cyclists,
            graph_type=experiment_noisy.graph_type,
            trip_type=experiment_noisy.trip_type,
            graph_file=["input", load, "$(load).graphml"],
            trips_file=["input", load, "$(load)_demand.json"],
            output_file=["output", save, "$(save)_data_$(mode_str).json"],
            log_file=["logs", "$(save)_$(mode_str).log"],
        )
        alg_experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode_str).json")
        BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
        BikePathNet.run_simulation(alg_experiment_file)

        res = load_core_algorithm_result(datadir("output", save, "$(save)_data_$(mode_str).json"))

        bp_order_file = datadir("output", save, "$(save)_order_$(mode_str).json")
        open(bp_order_file, "w") do io
            write(io, to_json(res.edited_edges))
        end
        utility_file = datadir("output", save, "$(save)_utility_$(mode_str).json")
        open(utility_file, "w") do io
            write(io, to_json(res.total_utility))
        end
        bpp_file = datadir("output", save, "$(save)_bpp_$(mode_str).json")
        open(bpp_file, "w") do io
            write(io, to_json(res.bike_path_percentage))
        end
        rm(datadir("output", save, "$(save)_data_$(mode_str).json"))
    end
end


for eval_save in eval_saves
    load = base_load
    save = base_save*eval_save

    for (mode_str, mode) in modes_noisy
        GC.gc()

        fixed_order = JSON.parsefile(datadir("output", save, "$(save)_order_$(mode_str).json"))

        eval_config = BikePathNet.AlgorithmConfig(
                buildup=config.buildup,
                use_existing_infrastructure=mode[2],
                use_blocked_streets=false,
                add_blocked_streets=[(-1, -1)],
                use_bike_highways=config.use_bike_highways,
                save_edge_loads=false,
                undirected=config.undirected,
                minmode=BikePathNet.FixedOrderMinmode(fixed_order),
                beta=config.beta,
            )

        eval_experiment = BikePathNet.Experiment(
            save=experiment.save,
            city_name=experiment.city_name,
            algorithm_config=eval_config,
            cyclists=experiment.cyclists,
            graph_type=experiment.graph_type,
            trip_type=experiment.trip_type,
            graph_file=["input", load, "$(load).graphml"],
            trips_file=["input", load, "$(load)_demand.json"],
            output_file=["output", save, "$(save)_data_$(mode_str)_eval.json"],
            log_file=["logs", "$(save)_$(mode_str).log"],
        )
        eval_experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode_str)_eval.json")
        BikePathNet.save_experiment(eval_experiment_file, eval_experiment)
        BikePathNet.run_simulation(eval_experiment_file)

        res = load_core_algorithm_result(datadir("output", save, "$(save)_data_$(mode_str)_eval.json"))

        bp_order_file = datadir("output", save, "$(save)_order_$(mode_str)_eval.json")
        open(bp_order_file, "w") do io
            write(io, to_json(res.edited_edges))
        end
        utility_file = datadir("output", save, "$(save)_utility_$(mode_str)_eval.json")
        open(utility_file, "w") do io
            write(io, to_json(res.total_utility))
        end
        bpp_file = datadir("output", save, "$(save)_bpp_$(mode_str).json")
        open(bpp_file, "w") do io
            write(io, to_json(res.bike_path_percentage))
        end
        rm(datadir("output", save, "$(save)_data_$(mode_str)_eval.json"))
    end
end