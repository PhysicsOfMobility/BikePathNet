using DrWatson
@quickactivate :BikePathNet
using JSON
using Serde

city = "Copenhagen"
base_save = "cph_tud"
param_saves = vcat([""], ["_noisy_demand_"*"$i" for i in 1:10], ["_noisy_costs_"*"$i" for i in 1:10], ["_noisy_speeds_"*"$i" for i in 1:10])
modes = [
    ("Penalty", BikePathNet.MinPenaltyLengthCostWeightedCyclists()), 
    ("CS", BikePathNet.ConsumerSurplusCost()), 
    ("CSHB", BikePathNet.ConsumerSurplusHealthBenefitsCost()),
    ]

for param_save in param_saves
    save = base_save*param_save

    params = Dict()

    cyclist_types = 9
    speed_conversion = 3.6
    speeds = JSON.parsefile(datadir("input", save, "$(save)_speeds.json"))
    super_speeds = speeds["super_speeds"] / speed_conversion
    bp_speeds = speeds["bp_speeds"] / speed_conversion
    street_speeds = speeds["street_speeds"] / speed_conversion
    
    params["speeds"] = super_speeds
    params["car_penalty"] = Dict([
                ("primary", street_speeds./super_speeds),
                ("residential", bp_speeds./super_speeds),
                ])
    params["slope_penalty"] = Dict([(0.0, ones(cyclist_types))])
    params["intersection_penalty"] = Dict([("large", zeros(cyclist_types)), ("medium", zeros(cyclist_types)), ("small", zeros(cyclist_types))])
    params["edge_time_penalty"] = Dict([("large", ones(cyclist_types)*30), ("medium", ones(cyclist_types)*5), ("small", zeros(cyclist_types))])
    params["turn_penalty"] = Dict([("large", zeros(cyclist_types)), ("medium", zeros(cyclist_types)), ("small", zeros(cyclist_types))])
    params["bp_end_penalty"] = Dict([
        ("primary", Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        ("residential", Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        ])
    
    beta = -0.05184046 / 60
    values_hb = [7.15983, 7.15983, 7.15983, 4.09945, 4.09945, 4.09945, 4.09945, 4.09945, 4.09945] / 1000
    
    fixed_order = []
    
    cyclists = BikePathNet.Cyclist[]
    for i in 1:cyclist_types
        push!(cyclists, BikePathNet.Cyclist(
            id=i,
            speed=params["speeds"][i],
            car_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["car_penalty"]),
            slope_penalty=Dict(k => v[i] for (k, v) in params["slope_penalty"]),
            intersection_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["intersection_penalty"]),
            edge_time_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["edge_time_penalty"]),
            turn_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["turn_penalty"]),
            bikepath_end_penalty=Dict(Symbol(k) => Dict(k_j => v_j[i] for (k_j, v_j) in v) for (k, v) in params["bp_end_penalty"]),
            value_hb=values_hb[i]
        ))
    end
    
    for mode in modes
        trip_type = if mode[1] == "Penalty" trip_type=BikePathNet.Trip else trip_type=BikePathNet.VariableTrip end
        algo_config = BikePathNet.AlgorithmConfig(
            buildup=false,
            use_existing_infrastructure=true,
            use_blocked_streets=true,
            add_blocked_streets=[(-1, -1)],
            use_bike_highways=true,
            save_edge_loads=false,
            undirected=false,
            minmode=mode[2],
            beta=beta,
        )
        experiment = BikePathNet.Experiment(
            save=save,
            city_name=city,
            algorithm_config=algo_config,
            cyclists=cyclists,
            graph_type=BikePathNet.SegmentGraph,
            trip_type=trip_type,
            graph_file=["input", save, "$(save).graphml"],
            trips_file=["input", save, "$(save)_demand.json"],
            output_file=["output", save, "$(save)_data_$(mode[1]).json"],
            log_file=["logs", "$(save)_$(mode[1]).log"],
        )
        experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode[1]).json")
    
        BikePathNet.save_experiment(experiment_file, experiment)
        BikePathNet.run_simulation(experiment_file)

        res = load_core_algorithm_result(datadir("output", save, "$(save)_data_$(mode[1]).json"))
        bp_order = reverse(unique!(res.edited_segments))
        bp_order_file = datadir("output", save, "$(save)_order_$(mode[1]).json")
        open(bp_order_file, "w") do io
            write(io, to_json(bp_order))
        end    
    end
end


eval_saves = vcat(["_noisy_demand_"*"$i" for i in 1:10], ["_noisy_costs_"*"$i" for i in 1:10], ["_noisy_speeds_"*"$i" for i in 1:10])
eval_modes = [
    "Penalty", 
    "CS", 
    "CSHB",
    ]

for eval_save in eval_saves
    output_save = base_save*eval_save

    params = Dict()

    cyclist_types = 9
    speed_conversion = 3.6
    speeds = JSON.parsefile(datadir("input", base_save, "$(base_save)_speeds.json"))
    super_speeds = speeds["super_speeds"] / speed_conversion
    bp_speeds = speeds["bp_speeds"] / speed_conversion
    street_speeds = speeds["street_speeds"] / speed_conversion
    
    params["speeds"] = super_speeds
    params["car_penalty"] = Dict([
                ("primary", street_speeds./super_speeds),
                ("residential", bp_speeds./super_speeds),
                ])
    params["slope_penalty"] = Dict([(0.0, ones(cyclist_types))])
    params["intersection_penalty"] = Dict([("large", zeros(cyclist_types)), ("medium", zeros(cyclist_types)), ("small", zeros(cyclist_types))])
    params["edge_time_penalty"] = Dict([("large", ones(cyclist_types)*30), ("medium", ones(cyclist_types)*5), ("small", zeros(cyclist_types))])
    params["turn_penalty"] = Dict([("large", zeros(cyclist_types)), ("medium", zeros(cyclist_types)), ("small", zeros(cyclist_types))])
    params["bp_end_penalty"] = Dict([
        ("primary", Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        ("residential", Dict([(1, zeros(cyclist_types)), (2, zeros(cyclist_types)), (3, zeros(cyclist_types))])),
        ])
    
    beta = -0.05184046 / 60
    values_hb = [7.15983, 7.15983, 7.15983, 4.09945, 4.09945, 4.09945, 4.09945, 4.09945, 4.09945] / 1000
    
    cyclists = BikePathNet.Cyclist[]
    for i in 1:cyclist_types
        push!(cyclists, BikePathNet.Cyclist(
            id=i,
            speed=params["speeds"][i],
            car_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["car_penalty"]),
            slope_penalty=Dict(k => v[i] for (k, v) in params["slope_penalty"]),
            intersection_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["intersection_penalty"]),
            edge_time_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["edge_time_penalty"]),
            turn_penalty=Dict(Symbol(k) => v[i] for (k, v) in params["turn_penalty"]),
            bikepath_end_penalty=Dict(Symbol(k) => Dict(k_j => v_j[i] for (k_j, v_j) in v) for (k, v) in params["bp_end_penalty"]),
            value_hb=values_hb[i]
        ))
    end
    
    for mode in eval_modes
        fixed_order = JSON.parsefile(datadir("output", output_save, "$(output_save)_order_$(mode).json"))
        fixed_order = reverse(fixed_order)

        algo_config = BikePathNet.AlgorithmConfig(
            buildup=false,
            use_existing_infrastructure=true,
            use_blocked_streets=true,
            add_blocked_streets=[(-1, -1)],
            use_bike_highways=true,
            save_edge_loads=false,
            undirected=false,
            minmode=BikePathNet.FixedOrderMinmode(fixed_order),
            beta=beta,
        )
        experiment = BikePathNet.Experiment(
            save=output_save,
            city_name=city,
            algorithm_config=algo_config,
            cyclists=cyclists,
            graph_type=BikePathNet.SegmentGraph,
            trip_type=BikePathNet.VariableTrip,
            graph_file=["input", base_save, "$(base_save).graphml"],
            trips_file=["input", base_save, "$(base_save)_demand.json"],
            output_file=["output", output_save, "$(output_save)_data_$(mode)_eval.json"],
            log_file=["logs", "$(output_save)_$(mode).log"],
        )
        experiment_file = datadir("input", output_save, "$(output_save)_algorithm_params_$(mode)_eval.json")
    
        BikePathNet.save_experiment(experiment_file, experiment)
        BikePathNet.run_simulation(experiment_file)

        res = load_core_algorithm_result(datadir("output", output_save, "$(output_save)_data_$(mode)_eval.json"))
        bp_order = reverse(unique!(res.edited_segments))
        bp_order_file = datadir("output", output_save, "$(output_save)_order_$(mode)_eval.json")
        open(bp_order_file, "w") do io
            write(io, to_json(bp_order))
        end
        utility_file = datadir("output", output_save, "$(output_save)_utility_$(mode)_eval.json")
        open(utility_file, "w") do io
            write(io, to_json(res.total_utility))
        end
    end
end