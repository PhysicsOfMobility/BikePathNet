"""
This type encapulates the individual properties of a cyclist such as `speed` and
various penalties for different road- and intersection-types. All multiplicative penalties are unitless, all  additive penalties are measured in seconds.
"""
@kwdef struct Cyclist
    "unique identifier for the cyclist"
    id::Int
    "speed of the cyclist in m/s"
    speed::Float64
    "multiplicative penalty for using unprotected streets, indexed by the `street_type` of [`Edge`](@ref). (see [`car_penalty`](@ref))"
    car_penalty::Dict{Symbol,Float64}
    "multiplicative penalty for the slope of streets. (see [`slope_penalty`](@ref))"
    slope_penalty::Dict{Float64,Float64}
    "additive penalty for the size of intersections. Indexed by `size` of [`Node`](@ref). (see [`intersection_penalty`](@ref))"
    intersection_penalty::Dict{Symbol,Float64}
    "additive penalty for the time it takes to traverse a street. Indexed by `time_penalty` of [`Edge`](@ref). (see [`edge_time_penalty`](@ref))"
    edge_time_penalty::Dict{Symbol,Float64}
    "additive penalty for turning from one edge to another. Indexed by the `turn_badness` of the [`Edge`](@ref) we turn on. (see [`turn_penalty`](@ref))"
    turn_penalty::Dict{Symbol,Float64}
    "additive penalty for cycling of a bike path, onto a unprotected street. Indexed by the `street_type` of the [`Edge`](@ref) we turn on, as well as the number of times the cyclist left a bike lane already. (see [`bike_path_end_penalty`](@ref))"
    bikepath_end_penalty::Dict{Symbol,Dict{Int,Float64}}
    "value of health benefits (price per unit length)"
    value_hb::Float64
end

function Base.show(io::IO, ::MIME"text/plain", config::T) where {T<:Cyclist}
    println(io, T)
    for field in fieldnames(T)
        println(io, "  ", field, ": ", getfield(config, field))
    end
end

######## How to write a Cyclist to json #######
function Serde.SerJson.ser_value(::Type{Cyclist}, ::Val{:slope_penalty}, v)
    return Dict(string(k) => v for (k, v) in v)
end

function Serde.SerJson.ser_value(::Type{Cyclist}, ::Val{:bikepath_end_penalty}, v)
    return Dict(k => Dict(string(k2) => v2 for (k2, v2) in v) for (k, v) in v)
end