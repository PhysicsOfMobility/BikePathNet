# Installation

There are two options how you can use the code in this package:

## Option 1: Using the Package as a Dependency
If you do not want to modify or access the codebase, you can simply add the package as a dependency to your project by running the following code with your current project being active in the julia repl:
```julia
julia> using Pkg
julia> Pkg.add(PackageSpec(url="https://github.com/christophsteinacker/BikePathNet"; rev="main"))
```
this will add the package as it exists on the `main` branch to your project and you can use it as a normal julia package with:
```julia
julia> using BikePathNet
julia> load_graph("path/to/graph", SegmentGraph)
```
## Option 2: Cloning the Repository
If you want to have direct access to all the code and the possibility to modify it, you can clone the repository and work directly within it:

```shell
git clone https://github.com/christophsteinacker/BikePathNet
```
Then, navigate to the folder, open a Julia REPL and run the following commands:
```julia
julia> using Pkg
julia> Pkg.add("DrWatson") # install globally, for using `@quickactivate`
julia> Pkg.activate("path/to/this/project")
julia> Pkg.instantiate()
```
After this, you can write your scripts in the `/scripts/` folder, edit the package code under `/src/` an produce your own plots and results.
