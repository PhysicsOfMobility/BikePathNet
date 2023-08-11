# BikePathNet

This project is designed to help to improve the bikeability of cities. 
Therefore, it combines a route choice model for the cyclists, based on street size and presence or absence of bike paths along streets, OpenStreetMap data and the cyclists demand.
## Setup Linux
1. ...
## Setup Windows
1. Create conda environment with Python 3.10 (named python310 from here on)
2. Install julia
3. Install julia packages from julia REPL (in Julia REPL press <kbd>]</kbd>)
   ```
   pkg> add HDF5 JSON Dates LoggingExtras DataStructures PyCall DrWatson
   ```
4. Rebuild PyCall for python310 interpreter in julia:
   ```julia
   julia> ENV["PYTHON"] = raw"\Path\to\conda\python310\python.exe"
   pkg> build PyCall
   ```
5. Install Python packages with pip (into python310)
   ```bash
   $ pip install -r requirements.txt
   ```
6. Install PyJulia in python310
   ```bash
   $ pip install --user julia
   ```
7. Launch a Python REPL in python310 and run
   ```python
   >>> import julia
   >>> julia.install()
   ```
   Check during install, if Python interpreter used is from python310.
8. Configure VS Code (if necessary):
   Set Conda Path to ```\Path\to\conda.exe``` and default interpreter path to ```\Path\to\conda\python310\python.exe```.
   Enable ```Execute In File Dir``` in the settings.

## DrWatson Setup (replace windows/linux setup with this later on)

To (locally) reproduce this project, do the following:

Open a Julia console and do:
   ```
   pkg> activate("path/to/this/project")
   pkg> instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson

@quickactivate "BikePathNet"
```
which auto-activate the project and enable local path handling from [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/).
