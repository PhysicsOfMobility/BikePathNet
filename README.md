# BikePathNet

This project is designed to help to improve the bikeability of cities. 
Therefore, it combines a route choice model for the cyclists, based on street size and presence or absence of bike paths along streets, OpenStreetMap data and the cyclists demand.

## Setup
The setup is written for Linux/macOS systems. For Windows systems some commands might be slightly different, but the general setup structure stays the same.
You need Python in version 3.12 and Julia in version 1.10 installed.
1. Create a Python virtual environment (named venv from here on) and add all packages from the `requirements.txt`.
   ```bash
   python3 -m venv /path/for/venv
   source /path/to/venv/activate
   pip install -r requirements.txt
   ```
2. Install julia packages via the julia REPL (in Julia REPL press <kbd>]</kbd>)
   ```julia
   pkg> add DrWatson
   pkg> activate("path/to/this/project")
   pkg> instantiate()
   ```
3. Optional: If you want to include elevation/slope data for the algorithm you need to download world elevation data provided by the [NASA Shuttle Radar Topography Mission (SRTM)](https://www2.jpl.nasa.gov/srtm/). This project only utilises the more precise 1 arc-seconds (30 meters) data. After downloading the data add the location to the environment variables.
   ```bash
   export SRTM1_DIR=/path/to/srtm1_data/
   ```
   If you don't want to use the elevation data you can skip this step, all node elevations will be set to 0.
   
## Running the calculations

All commands assume you are in the project folder.
1. Prepare data for algorithm with given street networks.
   ```bash
   python3 examples/hh_prep.py
   ```
   It will take only a couple of seconds.
2. Run the simulations.
   ```bash
   julia examples/hh_algorithm.jl
   ```
   If you want to make use of parallelizing parts of the calculations, specify the number of threads (e.g. 8) available to julia.
   ```bash
   julia --threads 8 examples/hh_algorithm.jl
   ```
   The calculations take 5-15 minutes depending on the given number of CPU cores and their speed. 
3. Now the figures can be generated, some additional information will be printed in the commandline.
   ```bash
   python3 examples/hh_plot.py
   ```
   The plots will be saved in the `examples/plots/` folder. Should take less than 5 minutes for all plots.

### DrWatson
You may notice that the julia script start with the commands:
```julia
using DrWatson
@quickactivate "BikePathNet"
```
which auto-activate the project and enable local path handling from [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).


### Software Versions Used
1. Python 3.12
2. Julia 1.10.3
3. Python packages: See requirements.txt
4. PROJ 8.1.1


## Data Licenses
The data used for Hamburg (hh) in the example folder is extracted from a publicly available data set provided 
by [Call a Bike by Deutsche Bahn AG](https://data.deutschebahn.com/dataset/data-call-a-bike) (License: CC BY 4.0).