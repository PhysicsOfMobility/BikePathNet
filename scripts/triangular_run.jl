using DrWatson
using Printf
@quickactivate :BikePathNet
using Setfield
using JSON
using Serde

city = "Triangular"
base_save = "triangular"

cyclist_types = 1
params = Dict()
cyclist_types = cyclist_types
params["speeds"] = ones(cyclist_types)
params["car_penalty"] = Dict{Symbol,Vector{Float64}}(
    :primary => ones(cyclist_types) * 1/7,
    )
params["slope_penalty"] = Dict(
    0.0 => ones(cyclist_types)
    )
params["intersection_penalty"] = Dict(
    :medium => ones(cyclist_types) * 0.1, 
    :small => ones(cyclist_types) * 0.01
    )
params["turn_penalty"] = Dict(
    :medium => zeros(cyclist_types), 
    :small => zeros(cyclist_types)
    )
params["edge_time_penalty"] = Dict{Symbol,Vector{Float64}}([
    (:large, zeros(cyclist_types)), 
    (:medium, zeros(cyclist_types)), 
    (:small, zeros(cyclist_types))
    ])
params["bp_end_penalty"] = Dict{Symbol,Dict{Int,Vector{Float64}}}(:primary => Dict(1 => zeros(cyclist_types)))

params["surrogate"] = true
params["bike_highways"] = true

base_cyclists = BikePathNet.Cyclist[]
for i in 1:cyclist_types
    push!(base_cyclists, BikePathNet.Cyclist(
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
    use_blocked_streets=true,
    add_blocked_streets=[(-1, -1)],
    use_bike_highways=true,
    save_edge_loads=false,
    undirected=false,
    minmode=BikePathNet.PureCyclists(),
    beta=0.0,
    )
experiment = BikePathNet.Experiment(
    save=base_save,
    city_name=city,
    algorithm_config=config,
    cyclists=base_cyclists,
    graph_type=BikePathNet.Graph,
    trip_type=BikePathNet.Trip,
    graph_file=["input", base_save, "$(base_save).graphml"],
    trips_file=["input", base_save, "$(base_save)_demand.json"],
    output_file=["output", base_save, "$(base_save)_data.json"],
    log_file=["logs", "$(base_save).log"],
    )
modes = Dict(
    "000" => BikePathNet.CyclistsLengthNormed(),
    "010" => BikePathNet.NormalizedTravelTimePenalty(),
    "020" => BikePathNet.TotalTravelTimeEstimation(),
    "030" => BikePathNet.TotalTravelTimeEstimationPerLength(),
    "040" => BikePathNet.TotalTravelTimeEstimationCapped(),
    "050" => BikePathNet.TotalTravelTimeEstimationCappedPerLength(),
    "060" => BikePathNet.TotalDetour(),
    "070" => BikePathNet.TotalDetourPerLength(),
    "080" => BikePathNet.FixedOrderMinmode([-1]),
    "090" => BikePathNet.TotalTravelTimeEstimationUncapped(),
    )

for (mode_str, mode) in modes
    alg_config = @set config.minmode = mode
    alg_experiment = BikePathNet.Experiment(
        save=save,
        city_name=experiment.city_name,
        algorithm_config=alg_config,
        cyclists=experiment.cyclists,
        graph_type=experiment.graph_type,
        trip_type=experiment.trip_type,
        graph_file=["input", save, "$(save).graphml"],
        trips_file=["input", save, "$(save)_demand.json"],
        output_file=["output", save, "$(save)_data_$(mode_str).json"],
        log_file=["logs", "$(save)_$(mode_str).log"],
        )
    alg_experiment_file = datadir("input", save, "$(save)_algorithm_params_$(mode_str).json")
    BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
    BikePathNet.run_simulation(alg_experiment_file)
end

realisations = 0

for nw_save in nw_saves
    save = base_save*nw_save
    penalties = JSON.parsefile(datadir("input", save, "$(save)_penalties.json"))

    for n in 1:realisations
        for (mode_str, mode) in modes
            if mode_str == "080"
                fixed_order = JSON.parsefile(datadir("input", save, "$(save)_$(lpad(n, 2, "0"))_random_order.json"))
                alg_config = @set config.minmode = BikePathNet.FixedOrderMinmode(fixed_order)
            else
                alg_config = @set config.minmode = mode
            end
            
            alg_experiment = BikePathNet.Experiment(
                save="$(save)_realisation_$(lpad(n, 2, "0"))",
                city_name=experiment.city_name,
                algorithm_config=alg_config,
                cyclists=experiment.cyclists,
                graph_type=experiment.graph_type,
                trip_type=experiment.trip_type,
                graph_file=["input", save, "$(save).graphml"],
                trips_file=["input", save, "$(save)_demand.json"],
                output_file=["output", save, "$(save)_$(lpad(n, 2, "0"))_data_$(mode_str).json"],
                log_file=["logs", "$(save)_$(mode_str).log"],
                )
            alg_experiment_file = datadir("input", save, "$(save)_$(lpad(n, 2, "0"))_algorithm_params_$(mode_str).json")
            BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
            BikePathNet.run_simulation(alg_experiment_file)
        end
    end
end

car_penalty_range = 8.0:1.0:10.0
intersection_penalty_range = 0.0:0.1:1.0
car_inter_pen_realisations = 10

for nw_save in nw_saves
    pen_base_save = base_save*nw_save
    pen_save_folder = pen_base_save*"_pen_sweep"
    for c_pen in car_penalty_range
        c_pen_str = @sprintf "%2.3f" c_pen
        for i_pen in intersection_penalty_range
            i_pen_str = @sprintf "%2.3f" i_pen

            cyclists = BikePathNet.Cyclist[]
            for i in 1:cyclist_types
                push!(cyclists, BikePathNet.Cyclist(
                    id=i,
                    speed=base_cyclists[i].speed,
                    car_penalty=Dict{Symbol, Float64}(:primary =>1/c_pen),
                    slope_penalty=base_cyclists[i].slope_penalty,
                    intersection_penalty=Dict{Symbol, Float64}(:medium => i_pen, :small => i_pen/10),
                    edge_time_penalty=base_cyclists[i].edge_time_penalty,
                    turn_penalty=base_cyclists[i].turn_penalty,
                    bikepath_end_penalty=base_cyclists[i].bikepath_end_penalty,
                    value_hb=1.0
                ))
            end
            for n in 1:car_inter_pen_realisations
                save = "$(pen_base_save)_$(replace(c_pen_str, "." => "-"))_$(replace(i_pen_str, "." => "-"))_$(lpad(n, 2, "0"))"

                for (mode_str, mode) in modes
                    if mode_str == "080"
                        fixed_order = JSON.parsefile(datadir("input", pen_base_save, "$(pen_base_save)_$(lpad(n, 2, "0"))_random_order.json"))
                        alg_config = @set config.minmode = BikePathNet.FixedOrderMinmode(fixed_order)
                    else
                        alg_config = @set config.minmode = mode
                    end
                    alg_experiment = BikePathNet.Experiment(
                        save=save,
                        city_name=experiment.city_name,
                        algorithm_config=alg_config,
                        cyclists=cyclists,
                        graph_type=experiment.graph_type,
                        trip_type=experiment.trip_type,
                        graph_file=["input", pen_base_save, "$(pen_base_save).graphml"],
                        trips_file=["input", pen_base_save, "$(pen_base_save)_demand.json"],
                        output_file=["output", pen_save_folder, "$(save)_data_$(mode_str).json"],
                        log_file=["logs", "$(save)_$(mode_str).log"],
                        )
                    alg_experiment_file = datadir("input", pen_save_folder, "$(save)_$(lpad(n, 2, "0"))_algorithm_params_$(mode_str).json")
                    BikePathNet.save_experiment(alg_experiment_file, alg_experiment)
                    BikePathNet.run_simulation(alg_experiment_file)
                end
            end
        end
    end
end