[![DOI](https://zenodo.org/badge/497852121.svg)](https://zenodo.org/badge/latestdoi/497852121)

# Demand-driven design of bicycle infrastructure networks for improved urban bikeability

This is the source code for the paper "[Demand-driven design of bicycle infrastructure networks for improved urban bikeability](https://www.nature.com/articles/s43588-022-00318-w)" by C. Steinacker, D.-M. Storch, M. Timme, and M. Schröder.

## HOWTO

I strongly recommend using Linux (e.g. openSUSE Tumbleweed). With Windows, there are major problems with Conda and R, which means that the code often cannot be executed there.

1. Install the dependencies

   There are two options to install the dependencies: Conda or pure Python virtualenv. Both should not take more than 20-25 minutes to set up.
   1. Conda

      Check that you have Latex installed outside of conda, as the conda version of Latex is currently broken. Use the `conda-env.yml` file to create a conda environment: `conda env create -f conda-env.yml`. The environment will be called `BikePathNet`, activate the environment with `conda activate BikePathNet`.

   2. Python virtualenv
   
      Check if you have installed the following dependencies
      * Python 3 (>= 3.6) including development libraries
      * PROJ (>= 8.0)
      * R (>=3.6.1)
      * g++ (>= 5.3)
      * cmake (>= 3.5)
      * Build System: Make or Ninja
      * Latex
   
      Create Python venv and install Python requirements.
      ````bash
      # Create a new virtualenv and activate it
      python3 -m venv venv
      source venv/bin/activate
      # Install dependencies
      pip install cython
      pip install -r requirements.txt
      ````


2. Install the package

   Make sure you activated the conda env or Python virtualenv, and you are executing the command from the project folder.
   The installation of the package should only take a couple of seconds.
   ````bash
   pip install -e ./
   ````

3. Executing the python scripts
   
   All commands assume you are in the project folder.
   1. Prepare data for algorithm with given street networks. Execute `examples/hh_prep.py`. It will take only a couple of seconds.
   2. Run the simulations. This can be done by running `examples/hh_algorithm.py`. This is a time and resource intensive step, it takes roughly 1-2 hours depending on the number of CPU cores and their speed. 
   3. Now the figures can be generated, some additional information will be printed in the commandline. Execute `examples/hh_plot.py`. The plots will be saved in the `examples/plots/` folder. Should take less than 5 minutes for all plots.


### Software Versions Used
   1. Python 3.9
   2. Python packages: See requirements.txt
   3. R 3.6.1
   4. PROJ 8.1.1


## Data Licenses
The data used for Hamburg (hh) in the example folder is extracted from a publicly available data set provided 
by [Call a Bike by Deutsche Bahn AG](https://data.deutschebahn.com/dataset/data-call-a-bike) (License: CC BY 4.0).
