mutable struct Trip{T}
    origin :: T
    destination :: T
    nbr_of_users :: Float64
    nodes :: Vector{T}
    edges :: Vector{Tuple{T, T}}
    real_length :: Float64
    felt_length :: Float64
    real_length_on_types :: Dict{String,Float64}
    on_street :: Bool
    cyclist_type :: Int
end


function Trip{T}(origin::T, destination::T) where T
    return Trip{T}(origin, destination, 0.0, Vector{Int}(), Vector{Tuple{Int,Int}}(), 0.0, 0.0, Dict{String,Float64}(), false, 1)
end


function Trip(origin, destination)
    return Trip{typeof(origin)}(origin, destination)
end