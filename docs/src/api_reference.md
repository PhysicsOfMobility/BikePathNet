# API Reference

```@index
Pages = ["api_reference.md"]
```

## Graphs
```@autodocs
Modules = [BikePathNet]
Pages = ["graph.jl", "graph_loading.jl", "graph_measures.jl", "graph_preparation.jl"]
```

## Cyclists and Trips
```@autodocs
Modules = [BikePathNet]
Pages = ["cyclist.jl", "trip.jl", "induced_demand.jl"]
```

## Algorithm Configuration and Experiments
```@autodocs
Modules = [BikePathNet]
Pages = ["algorithm_config.jl", "experiment.jl", "params_io.jl"]
```

## Minmodes
```@docs
BikePathNet.AbstractMinmode
BikePathNet.AbstractMinmodeState
```
```@autodocs
Modules = [BikePathNet]
Pages = ["minmodes.jl", "loads_on_streets.jl"]
```

## Shortest Paths
```@docs
BikePathNet.AbstractAggregator
```
```@autodocs
Modules = [BikePathNet]
Pages = ["penalties.jl", "shortest_paths.jl"]
```

## Core Algorithms
```@autodocs
Modules = [BikePathNet]
Pages = ["algorithm.jl", "comparison_state.jl"]
```

## Logging
```@autodocs
Modules = [BikePathNet]
Pages = ["logging.jl"]
```

## BikePathNet
```@docs
BikePathNet
```