"""
This type collects all the configuration-options for the main algorithm
(see [`core_algorithm`](@ref) and [`calculate_comparison_state`](@ref)).
"""
@kwdef struct AlgorithmConfig{T<:AbstractMinmode}
    "`true` if the bike paths should be built up from scratch, `false` if it should be removed."
    buildup::Bool
    "`true` if the existing bike infrastructure should be preserved, `false` if it should be overwritten."
    use_existing_infrastructure::Bool
    "`true` if blocked streets (streets on which no bike path is allowed to be built)
    should be preserved, `false` if they should be overwritten."
    use_blocked_streets::Bool
    "Additional blocked streets."
    add_blocked_streets::Vector{Tuple{Int, Int}}
    "`true` if bike highways should be used, `false` otherwise."
    use_bike_highways::Bool  # TODO: what was this for?
    "`true` if edge loads should be saved in the result of the algorithm, `false` otherwise."
    save_edge_loads::Bool
    "`true` if the graph should be treated as undirected, `false` if it should be kept directed."
    undirected::Bool  # TODO: is this really implemented?
    "mode by which the algorithm chooses to find the next edge to add/remove (see [`AbstractMinmode`](@ref))."
    minmode::T
    "marginal utility of travel time for bikes which is used with [`VariableTrip`](@ref)"
    beta::Float64
end

function Base.show(io::IO, ::MIME"text/plain", config::T) where {T<:AlgorithmConfig}
    println(io, T)
    for field in fieldnames(T)
        println(io, "  ", field, ": ", getfield(config, field))
    end
end

# MARK: SerDe
function Serde.SerJson.ser_type(::Type{<:AlgorithmConfig}, v::T) where {T<:AbstractMinmode}
    minmode = split(split(string(T), '.')[end], "{")[1]
    data = Dict(String(f_name) => getfield(v, f_name) for f_name in fieldnames(T))
    return Dict("minmode" => minmode, "data" => data)
end

Serde.deser(::Type{<:AlgorithmConfig}, ::Type{<:AbstractMinmode}, data) = deser_minmode(Val(Symbol(data["minmode"])), data)

deser_minmode(::Val{:MinPureCyclists}, _) = MinPureCyclists()
deser_minmode(::Val{:MinPenaltyWeightedCyclists}, _) = MinPenaltyWeightedCyclists()
deser_minmode(::Val{:MinTotalDetour}, _) = MinTotalDetour()
deser_minmode(::Val{:MinPenaltyCostWeightedCyclists}, _) = MinPenaltyCostWeightedCyclists()
deser_minmode(::Val{:MinPenaltyLengthWeightedCyclists}, _) = MinPenaltyLengthWeightedCyclists()
deser_minmode(::Val{:MinPenaltyLengthCostWeightedCyclists}, _) = MinPenaltyLengthCostWeightedCyclists()
deser_minmode(::Val{:ConsumerSurplus}, _) = ConsumerSurplus()
deser_minmode(::Val{:ConsumerSurplusCost}, _) = ConsumerSurplusCost()
deser_minmode(::Val{:ConsumerSurplusHealthBenefits}, _) = ConsumerSurplusHealthBenefits()
deser_minmode(::Val{:ConsumerSurplusHealthBenefitsCost}, _) = ConsumerSurplusHealthBenefitsCost()
function deser_minmode(::Val{:FixedOrderMinmode}, data)
    if typeof(data["data"]["order"][1]) <: Int
        FixedOrderMinmode(convert(Vector{Int}, data["data"]["order"]))
    elseif typeof(data["data"]["order"][1]) <: Vector
        FixedOrderMinmode([Int.(tuple(edge...)) for edge in data["data"]["order"]])
    else
        println("Order has to be an array of segments (Int) or edges (Vector).")
    end
end

function Serde.deser(::Type{Vector{Tuple{Int64, Int64}}}, vector)
    return [Int.(tuple(edge...)) for edge in vector]
end