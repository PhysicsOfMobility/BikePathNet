# BikePathNet

This project is designed to help to improve the bikeability of cities. 
Therefore, it combines a route choice model for the cyclists, based on topography, street/intersection size and presence or absence of bike paths along streets, OpenStreetMap data and the cyclists demand.

## Setup
The setup is written for Linux/macOS systems. For Windows systems some commands might be slightly different, but the general setup structure stays the same.
You need Python compatible to version 3.12 and Julia compatible to version 1.10 installed.
1. Create a Python virtual environment (named venv from here on) and add all packages from the `requirements.txt`.
   ```bash
   python3 -m venv /path/for/venv
   source /path/to/venv/activate
   pip install -r requirements.txt
   ```
2. Install julia packages via the julia REPL (in Julia REPL press <kbd>]</kbd>)
   ```julia
   pkg> add DrWatson
   pkg> activate path/to/this/project
   pkg> instantiate
   ```
3. Optional: If you want to include elevation/slope data for the algorithm you can either provide an elevation raster file for your desired area. 
Or you can download the world elevation data provided by the [NASA Shuttle Radar Topography Mission (SRTM)](https://www2.jpl.nasa.gov/srtm/). This project only utilises the more precise 1 arc-seconds (30 meters) data. After downloading the data add the location to the environment variables.
   ```bash
   export SRTM1_DIR=/path/to/srtm1_data/
   ```
   
   If you don't want to use any elevation data you can skip this step, all node elevations will be set to 0.
   
## Running the calculations

This is exemplary for the dataset for Dresden (dd), if you want to use your own data, please use the corresponding notebooks and scrits as a guidline for your setup. All commands assume you are in the project folder and python/Jupyter runs inside the venv.
1. Prepare the data for all penalty data sets of Dresden.
   ```bash
   python scripts/dd_prep.py
   ```
2. Run the simulation. Check beforehand if you really want to run the simulations for all criteria and data sets, this can take very long.
   ```bash
   julia scripts/dd_run.jl
   ```
   If you want to make use of parallelizing parts of the calculations, specify the number of threads (e.g. 8) available to julia.
   ```bash
   julia --threads 8 scripts/dd_run.jl
   ```
   The calculations take a couple of hours and greatly benefits from multithreading. 
3. Figures and other evaluation can now be generated in the Jupyter notebook `dd_plot.ipynb` in the `notebooks` folder. The Jupyter server should be started from inside the python vev. The plots will be saved in the `plots` folder.

### DrWatson
You may notice that the julia script start with the commands:
```julia
using DrWatson
@quickactivate "BikePathNet"
```
which auto-activate the project and enable local path handling from [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).


### Software Versions Used
1. Python 3.12.3
2. Julia 1.10.0
3. Python packages and dependencies: See requirements.txt
4. Julia packages and dependencies: See Mainfest.toml


## Data Licenses
The demand data is extracted from Kontur population data sets for [Germany](https://data.humdata.org/dataset/kontur-population-germany) and the [USA](https://data.humdata.org/dataset/kontur-population-united-states-of-america), distributed under [Creative Commons Attribution International (CC BY) license](https://creativecommons.org/licenses/by/4.0).
The graphs are heavily based on OpenStreetMap (OSM) data distributed under [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1.0/).
The [elevation data for Dresden]((https://opendata.dresden.de/informationsportal/?open=1&result=9F7A6477F63840F791F7CCD50C6CD9CF#app/mainpage)) is published by the city of Dresden under [dl-de/by-2-0 license](https://www.govdata.de/dl-de/by-2-0).