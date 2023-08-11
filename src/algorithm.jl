using JSON
using Dates


"""
    core_algorithm(G, trips, minmode=1, buildup=false, start_time=now(), save_load=False, cyclist_types=1)

Run the core algorithm for the given Graph 'G' and trips 'trips'. It is possible to choose between different minimal edge choosing modes,
    via 'minmode'. If the algorithm should start fro scratch set 'buildup' to false. 'start_time' is for logging purposes.
"""
function core_algorithm(G::Graph, trips::Dict{Int,Trip}, minmode=1::Int, buildup=false::Bool, start_time=now()::DateTime, save_edge_load=false::Bool, cyclist_types=1::Int; undirected::Bool=false)
    # Initial calculation
    @info "Initial calculation started."
    calc_trips!(G, trips, cyclist_types)
    @info "Initial calculation ended."

    # println([edge for (edge, edge_info) in G.edges if edge_info.load == 0.0])
    run_length = length(G.edges)

    # Initialise lists
    total_cost = Vector{Float64}([0.0])
    sizehint!(total_cost, run_length)
    bike_path_perc = [bike_path_percentage(G)]
    sizehint!(bike_path_perc, run_length)
    total_real_distance_traveled = Vector{Dict{String,Float64}}([total_real_dist_on_types(trips)])
    sizehint!(total_real_distance_traveled, run_length)
    total_felt_distance_traveled = Vector{Float64}([sum([trip.nbr_of_users * trip.felt_length for trip in values(trips)])])
    sizehint!(total_felt_distance_traveled, run_length)
    nbr_on_street = Vector{Float64}([nbr_of_trips_on_street(trips)])
    sizehint!(nbr_on_street, run_length)
    if save_edge_load
        edge_load = Dict{Tuple{Int,Int},Vector{Float}}([(edge, [sum(edge_info.load)]) for (edge, edge_info) in G.edges])
    else
        edge_load = 0
    end
    edited_edges = Vector{Tuple{Int, Int}}()
    sizehint!(nbr_on_street, run_length)
    used_ps_edges = Vector{Tuple{Int, Int}}()
    unused_edges = Vector{Tuple{Int, Int}}()
    cut = 0

    if !buildup
        log_at = Vector{Float64}([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01, 0, -1])
    else
        log_at = [i for i in range(0.1, 1.1, 11)]
    end
    next_log = popfirst!(log_at)

    first_used = true
    while true
        min_loaded_edge, used = get_minimal_loaded_edge(G, trips, minmode, buildup, cyclist_types, undirected=undirected)
        if min_loaded_edge == (-1, -1)
            break
        end
        if used && first_used
            used_ps_edges = get_used_primary_secondary(G)
            unused_edges = edited_edges
            cut = length(G.edges) - length(edited_edges)
            @info "First used bike path after $(length(G.edges) - cut) iterations."
            first_used = false
        end

        push!(edited_edges, min_loaded_edge)

        if !G.edges[min_loaded_edge].ex_inf
            # Calculate len of all trips running over min loaded edge.
            # Calculate cost of "adding" bike path
            push!(total_cost, G.edges[min_loaded_edge].cost)
            # Get all trips affected by editing the edge
            if buildup
                trips_recalc = deepcopy(trips)
            else
                trips_recalc = Dict{Int,Trip}([(trip_id, trips[trip_id]) for trip_id in G.edges[min_loaded_edge].trips])
            end
            # Edit minimal loaded edge and update graph.
            edit_edge!(G, min_loaded_edge, undirected=undirected)
            # Recalculate all affected trips and update their information.
            calc_trips!(G, trips_recalc, cyclist_types)
            trips = merge(trips, trips_recalc)
        else
            G.edges[min_loaded_edge].edited = true
            push!(total_cost, 0.0)
        end
        # Store all important data
        push!(bike_path_perc, bike_path_percentage(G))
        push!(total_real_distance_traveled, total_real_dist_on_types(trips))
        push!(total_felt_distance_traveled, sum([trip.nbr_of_users * trip.felt_length for trip in values(trips)]))
        push!(nbr_on_street, nbr_of_trips_on_street(trips))
        if save_edge_load
            set_edge_load!(G, edge_load)
        end

        if !buildup
            if (last(bike_path_perc) < next_log)
                @info "Reached $next_log BPP (removing) after $(canonicalize(round(now()-start_time, Dates.Second(1))))"
                next_log = popfirst!(log_at)
                @sync GC.gc()
            end
        else
            if (last(bike_path_perc) > next_log)
                @info "Reached $next_log BPP (buildup) after $(canonicalize(round(now()-start_time, Dates.Second(1))))"
                next_log = popfirst!(log_at)
                @sync GC.gc()
            end
        end
    end

    return edited_edges, bike_path_perc, total_cost, total_real_distance_traveled, total_felt_distance_traveled, nbr_on_street, edge_load, used_ps_edges, cut
end



"""
    run_simulation(city::String, save::String, input_folder::String, output_folder::String, params_file::String, log_folder::String)

Runs the algorithm for a given city ('city', 'save'). Loads all necessary data from the 'input_folder' and the parameters for the algorithm from the 'params_file'. 
    Saves results to the 'output_folder' and loggs event into a city and parameter specific file in the 'log_folder'.
"""
function run_simulation(city::String, save::String, input_folder::String, output_folder::String, params_file::String, log_folder::String)

    # Load parameters for the calculation
    mode, speeds, car_penalties, slope_penalties, surface_penalties, intersection_penalties, turn_penalties, bp_end_penalties, save_edge_load, cyclist_types, blocked = load_algorithm_params(params_file)

    # Check if output and log folder exist, otherwise creates them.
    mkpath(output_folder)
    mkpath(log_folder)

    buildup = mode[1]
    minmode = mode[2]
    ex_inf = mode[3]
    mode = "$(convert(Int, buildup))$(minmode)$(convert(Int, ex_inf))"
    
    setup_logger(joinpath(log_folder, "$(save)_$(mode).log"))

    start_time = now()
    @info "Starting $city with mode $(mode)."

    undirected = false
    G = load_graph(joinpath(input_folder, "$(save)_graph.json"), speeds, car_penalties, slope_penalties, surface_penalties, intersection_penalties, turn_penalties, bp_end_penalties, ex_inf, blocked, buildup, cyclist_types)
    @info "Loaded graph for $city with $(length(G.nodes)) nodes and $(length(G.edges)) edges."

    trips, stations = load_demand(joinpath(input_folder, "$(save)_demand.json"), cyclist_types)
    @info "Loaded $(sum([trip.nbr_of_users for (trip_id, trip) in trips])) ($(length(trips)) unique trips) trips between $(length(stations)) stations."

    edited_edges, bike_path_perc, total_cost, total_real_distance_traveled, total_felt_distance_traveled, nbr_on_street, edge_load, used_ps_edges, cut = 
        core_algorithm(G, trips, minmode, buildup, start_time, save_edge_load, cyclist_types, undirected=undirected)
    
    hf = h5open(joinpath(output_folder, "$(save)_data_mode_$(mode).hdf5"), "w")
    grp = create_group(hf, "all")
    grp["ee"] = JSON.json(edited_edges)
    grp["used_ps_edges"] = JSON.json(used_ps_edges)
    grp["cut"] = cut
    grp["bpp"] = bike_path_perc
    grp["cost"] = total_cost
    grp["trdt"] = JSON.json(total_real_distance_traveled)
    grp["tfdt"] = total_felt_distance_traveled
    grp["nos"] = nbr_on_street
    grp["edge_load"] = JSON.json(edge_load)
    close(hf)

    end_time = now()
    @info "Finished $city [$(mode)] after $(canonicalize(round(end_time-start_time, Dates.Second(1))))."

    @sync GC.gc()

    return nothing
end


"""
    calc_comparison_state(save::String, state::String, input_folder::String, output_folder::String, params_file::String)

Calculates the data for a comparison bike path network. If no bike paths are given, the trips are calculated in a bike path free graph.
"""
function calc_comparison_state(save::String, state::String, input_folder::String, output_folder::String, params_file::String; use_base_save_input::Bool=true)

    @info "Calculating $(state) comparison state."

    speeds, car_penalties, slope_penalties, surface_penalties, intersection_penalties, turn_penalties, bp_end_penalties, bike_paths, ex_inf, cyclist_types, blocked = load_comparison_state_params(params_file)

    G = load_graph(joinpath(input_folder, "$(save)_graph.json"), speeds, car_penalties, slope_penalties, surface_penalties, intersection_penalties, turn_penalties, bp_end_penalties, ex_inf, blocked, false, cyclist_types)
    
    for edge in keys(G.edges)
        if ex_inf
            if G.edges[edge].ex_inf
                if !(edge in bike_paths)
                    push!(bike_paths, edge)
                end
            end
        end
        if !(edge in bike_paths) && !G.edges[edge].blocked
            edit_edge!(G, edge)
        end
    end
    if use_base_save_input
        trips, stations = load_demand(joinpath(input_folder, "$(split(save, "_")[1])_demand.json"))
    else
        trips, stations = load_demand(joinpath(input_folder, "$(save)_demand.json"))
    end

    calc_trips!(G, trips, cyclist_types)

    total_cost = get_total_cost(G, bike_paths, ex_inf)
    bike_path_perc = bike_path_percentage(G)
    total_real_distance_traveled = total_real_dist_on_types(trips)
    total_felt_distance_traveled = sum([trip.nbr_of_users * trip.felt_length for trip in values(trips)])
    nbr_on_street = nbr_of_trips_on_street(trips)
    edge_load = Dict{Tuple{Int,Int},Float64}([(edge, sum(edge_info.load)) for (edge, edge_info) in G.edges])
    
    hf = h5open(joinpath(output_folder, "$(save)_data_comparison_state_$(state).hdf5"), "w")
    hf["bike_paths"] = JSON.json(bike_paths)
    hf["bpp"] = bike_path_perc
    hf["cost"] = total_cost
    hf["trdt"] = JSON.json(total_real_distance_traveled)
    hf["tfdt"] = total_felt_distance_traveled
    hf["nos"] = nbr_on_street
    hf["edge_load"] = JSON.json(edge_load)
    close(hf)

    return Dict([("bike_paths", bike_paths), ("bpp", bike_path_perc), ("cost", total_cost), ("trdt", total_real_distance_traveled), ("tfdt", total_felt_distance_traveled), ("nos", nbr_on_street), ("edge_load", edge_load)])
end


"""
    create_algorithm_params_file(save::String, params::Dict, paths::Dict, mode::Tuple{Bool,Int,Bool})

"""
function create_algorithm_params_file(save::String, params::Dict, paths::Dict, mode::Tuple{Bool,Int,Bool})
    input_folder = joinpath(paths["input_folder"], save)

    algorithm_params = Dict()
    algorithm_params["mode"] = mode
    algorithm_params["save_edge_load"] = params["save_edge_load"]
    algorithm_params["cyclist_types"] = params["cyclist_types"]
    algorithm_params["speeds"] = params["speeds"]
    algorithm_params["car_penalty"] = params["car_penalty"]
    algorithm_params["slope_penalty"] = params["slope_penalty"]
    algorithm_params["surface_penalty"] = params["surface_penalty"]
    algorithm_params["intersection_penalty"] = params["intersection_penalty"]
    algorithm_params["turn_penalty"] = params["turn_penalty"]
    algorithm_params["bp_end_penalty"] = params["bp_end_penalty"]
    algorithm_params["blocked"] = params["blocked"]

    open(joinpath(input_folder, "$(save)_algorithm_params_$(Int(mode[1]))$(mode[2])$(Int(mode[3])).json"),"w") do fp
        JSON.print(fp, algorithm_params)
    end
end

