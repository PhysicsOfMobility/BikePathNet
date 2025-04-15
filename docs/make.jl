using Documenter, DocumenterVitepress

using BikePathNet

getting_started_files = "getting_started/" .* [
    "0_installation.md",
    "1_quickstart.md",
    "2_tutorial.md",  # not sure what the difference between tutorial and quickstart is
]

documentation_files = "documentation/" .* [
    "0_experiments.md"
]

extending_files = "extending_bike_path_net/" .* [
    "1_minmodes.md",
    "2_shortest_path_aggregators.md"
]

makedocs(;
    modules=[BikePathNet],
    authors="Christoph Steinacker",
    repo="https://github.com/christophsteinacker/BikePathNet",
    sitename="BikePathNet.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/christophsteinacker/BikePathNet",
        devurl="dev",
        deploy_url="christophsteinacker.github.io/BikePathNet",
        build_vitepress=false
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => getting_started_files,
        "Documentation" => documentation_files,
        "Extending BikePathNet" => extending_files,
        "API Reference" => "api_reference.md"
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/christophsteinacker/BikePathNet",
    push_preview=true,
)
