mutable struct Edge{T}
    source :: T
    destination :: T
    weight :: Float64
    real_length :: Float64
    speed :: Vector{Float64}
    street_type :: String
    car_penalty :: Vector{Float64}
    slope :: Float64
    slope_penalty :: Vector{Float64}
    surface :: String
    surface_penalty :: Vector{Float64}
    turn_penalty :: Dict{T, Vector{Float64}}
    bp_end_penalty :: Dict{Int, Vector{Float64}}
    ex_inf :: Bool
    blocked :: Bool
    bike_path :: Bool
    cost :: Float64
    load :: Vector{Float64}
    trips :: Vector{Int}
    edited :: Bool
end