# **FultonMarket**
Fully integrated replica exchange simulation modules. 

## **Environment**
For preparation of a environment, please refer to [openmmtool installation instructions](https://openmmtools.readthedocs.io/en/stable/installation.html#).
Dependencies:
- openmmtools (0.25.2)
- mdtraj (1.11.0)
- scikit-learn (1.7.2)
- seaborn (0.13.2)
- jax ([see GPU instructions](https://docs.jax.dev/en/latest/installation.html))

## **Types of Simulation**

### *Parallel Tempering*

Parallel Tempering is powered by [openmmtools](https://openmmtools.readthedocs.io/en/stable/api/generated/openmmtools.multistate.ParallelTemperingSampler.html#openmmtools.multistate.ParallelTemperingSampler)

Description of Scripts:

- FultonMarket.py: Wrapper that runs shorter replica exchange simulations to properly manage disk space, storage considerations, and continuation of experiments. 
- Randolph.py: Powers the simulation. Includes options to add interpolation of thermodyanmic states to improve swapping.
- RUN_FULTONMARKET.py Executable to run/resume simulations. To see how to execute, type the following into the console.

```console
$ python RUN_FULTONMARKET.py --help
```

### *Parallel Tempering Analysis*
Analysis modules are built into this repo. 

Description of Scripts:

- FultonMarketAnalysis.py: Makes analysis objects to analyze and resample FultonMarket experiments. 
- RESAMPLE.py: Executable to perform importance resampling to acquire trajectories from FultonMarket experiments. To see how to execute, type the following into the console.

```console
$ python ./FultonMarket/analysis/RESAMPLE.py --help
```
    
        
