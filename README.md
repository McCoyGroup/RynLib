# RynLib

This started out as a quick layer between python and entos for running DMC

It's grown a bit...

## Overview

We provide an environment in which to run DMC with the potential of your choice with whatever little bells and whistles your little heart desires.

The system as designed has three segments to it, a general DMC package, a package for working with compiled potentials, and a package for managing MPI. 
The DMC package makes use of the compiled potentials and the MPI manager, using them to distribute its walkers to the various available cores and evaluating the energies.

The presence of so many moving parts, however, means that there are lots of knobs and levers available to tweak. 
Some of these, like the MPI, are set automatically, unless there's an override on the user side, but others, like the configuration of the potential, need user input.

Everything has been designed to run inside a container, which adds another layer of complexity.

## Getting Started

The first thing we have to do is actually install the package and build it out. All changes live on GitHub, so we'll start by cloning the repository

```ignorelang
git clone https://github.com/McCoyGroup/RynLib.git RynLib
```

If you just want to pull the latest updates, you can use `git pull`.

At this point, the specifics of the build process depend on whether you are on in a local compute environment in which case you'll want to use Docker, whether you're on a standard HPC in which case you're likely using Singularity, or if you're on NeRSC in which case you use Shifter. 

## Manual Build

Instructions for all of these environments are provided, with build files in the `setup` subdirectory of RynLib.

You can make this all work without a container, but it will be much more work and I'm not going to write up general instructions for that at this point.

In general, we build using `docker` in a local environment then push the container to [Docker Hub](https://hub.docker.com/). Integration with [Singularity Hub](https://singularity-hub.org/) might come at some point in the future, but for now we're just using docker.


## Running

After building out the image we can run the various exposed commands.

### Interface

Since this is happening inside a container, we provide a command-line interface to the packages inside. This looks like:

```ignorelang
[rynlib] group command [args]
```

where we have these groups and commands (this list is incomplete, but can get you started)

```ignorelang
config -- anything involved in configuring the overall package
    set_config CONFIG: sets the config file or options for RynLib
    configure_mpi: installs and compiles the necessary MPI libraries

sim -- anything involved in running the DMC itself, not in making the potential work
    list: lists the set of known DMC simulations
    status NAME: returns the status of the DMC simulation (timestep, number of walkers/wavefunctions, etc.)
    add NAME CONFIG: adds a simulation with the tag NAME to the set of known simulations and uses the file CONFIG
    run NAME: runs the simulation with the tag NAME
    restart NAME: removes any existing simulation data and restart the simulation with the tag NAME
    remove NAME: removes the simulation NAME
    
pot -- anything involved in configuring a potential for use in the DMC
    list: lists the set of known compiled potentials
    add NAME CONFIG SRC: adds a potential NAME to the set of known potentials using the file CONFIG and the source SRC
    status NAME returns the status of the potenial NAME
    compile NAME: attempts to compile the potential NAME if it has not already been
    remove NAME: removes the potential NAME
```

### The `rynlib` command

`rynlib` is provided as a general purpose Bash function in `setup/env.sh` along with some useful environment variables.

`RYNLIB_CONFIG_PATH` - where on the file-system to store results
`RYNLIB_ENTOS_PATH` - where on the file-system the `entos` folder extracted from an entos container lives (if we're using entos)

Update functions for Singularity and Shifter are also provided as `rynlib_update_singularity` and `rynlib_update_shifter`


Do keep in mind that with Shifter the `sbatch` process is [slightly different](https://docs.nersc.gov/programming/shifter/how-to-use/#running-jobs-in-shifter-images)

### Config Files

Most of the parts of the package (i.e. the simulations, potentials, and importance samplers) are set up using configuration files. 
There is a standardized configuration format for these files, which looks like

```python

# Preamble -- anything you need to get loaded
...

# Config dict

config = dict(
    param1=val,
    param2=val2,
    ...
)
```

This gets loaded in as a python module. 
To make this dynamically editable, all of the values in the `config` dict should be serializeable in a plain text format (think strings, lists of numbers, tuples, etc.). 
This could be relaxed in the future, but for now this just makes life way easier.

## Data

We're focused on the [Singularity](https://sylabs.io/docs/) and [Shifter](https://www.nersc.gov/research-and-development/user-defined-images/) use cases, but we'll also make it possible to use directly with Docker.

In all of these cases, we bind either a volume (Docker) or a directory on the host (Shifter and Singularity) to the container that it will write to.

The relevant bind is called `/config` by default, but this can be configured differently in the overall `config.py` file for the container if we want to put (e.g.) our simulation data elsewhere.
All data will be written to this volume, including the simulation data, importance samplers, potential data, and the primary `config.py`.

Since Docker mounts a volume, not a host directory, it can be a pain to get data out. 
For that, [this Stack Overflow answer](https://stackoverflow.com/a/35410781/5720002) is relevant.
For our use case this might look like

```ignorelang
tmp=$(docker run -d --mount source=simdata,target=/config -it rynimg --ignore)
docker cp $tmp:/config/<RELEVANT-DATA> <TARGET-DIR>
docker rm $tmp
```

This provides a persistence strategy, as by mounting a new volume you can change the configuration environment. For the most part, though, there should be no issue with always using a single volume.

For Shifter and Singularity the data gets written directly to the host, so it's easy to access.

## MPI

Working with MPI can also be a little subtle.
In this case, there are two variables you can set in the Dockerfile/Singularity definition file, `mpi_version` and `mpi_implementation`. 
These both have to be aligned with the environment you're working on.

For instance, on Hyak the default is to use OpenMPI v3.1.4 and so you need to set `mpi_version=3.1.4` and `mpi_implementation=ompi`. 
On NeRSC this is slightly different, as the `mpi_implementation=mpich`.

By default, this has already been set up for you, but it's worth keeping in mind in case you need to modify.

## Setting up a Potential

### Entos

By default, the RynLib container is built off of an Entos container and thus makes Entos internally available. 
To use it, we have to first configure the potential, by calling

```bash
rynlib pot configure-entos
```

and then we can test that this worked via

```bash
rynlib pot test-entos
```

### Directly Compiling the Potential

Since we need to call the potential inside the container environment, we need a way to compile the potential inside the container.

As an example, let's imagine we want to work with a harmonic oscillator. Maybe we have the source as `HarmonicOscillator.cpp`

```c++
double HarmonicOscillator(std::vector<std::vector<double> > coords, std::vector<std::string> atoms, double force_constant, double re) {
    // compute HO from coords, force_constant, and re
    ...
    return pot;
}
```

Already there's something to note, as _by default_ we pass the `atoms` in.
This means that either we need to do some more configuration work or we need to have that be an ignored parameter in our function call.

Now we need to decide how we want this to compile, which we set up by either creating a `Makefile` or a file called `build.sh`. Inside the container or other other of these will be called. 
By default, the code expects that this will make a library called `libHarmonicOscillator.so`, but if this isn't the case we can specify the name in our configuration file.
The only available compiler is `gcc/g++` so be aware of this when writing your build script.

### Using f2py

Because [f2py]() is such a common use case, we added support for calling into a general python potential. 
This requires an extra layer of indirection and so will be somewhat slower than using a directly compiled potential, but the convenience might make up for the slowdown. 

In this case, again, we provide a `Makefile`, but this file will call into `f2py` and return a proper python extension library.

Then we will write a package on the _python_ side that will load our potential from this and which will be defined like

```python
def _potential(walkers, atoms, extra_args):
    """
    :param walkers: the walkers to evaluate the potential over
    :type walkers: np.ndarray
    :param atoms: the atom symbols for the walker coordinates
    :type atoms: list[str]
    :param extra_args: a tuple of any extra ints, floats, and bools passed in
    :type extra_args: tuple
    """
    ...
```

the name _does_ have to be `_potential`, but if people also think it's really ugly it could definitely be changed

All of the data will get farmed out to the different cores when doing this.

### Using a Precompiled Binary

If you know the binary you have will work with the container, you can pass it in directly. In this case we simply have to turn off the `requires_make` flag.

### Loading the Potential

When getting a potential into the container, we use the command

```ignorelang
rynlib pot add src
```

The _source_ should be where our potential is stored, so either the directory containing the source or the potential binary.

The _config\_file_ holds the parameters for the potential. 
For the most part it just makes sense to look at one of the examples, since there are an overwhelming number of options.
At some point in the future, once the API has stabilized, I'll go back and document the options here.

## Setting up a Simulation

This will feel much like setting up a potential, but maybe a little bit simpler. 
We'll pass a directory (`SRC`) that contains a `config.py` file to set up the simulation as well as any other data that we might need to load in, e.g. a set of initial walkers.

```ignorelang
rynlib sim add NAME SRC
```

## Setting up Importance Sampling

An implementation of importance sampling is baked into the package, but this requires a user-side function to evaluate the trial wavefunction.
To make the config files as stateless as possible and to make it possible to use the same trial wavefunction over different simulation instances (think using 3000 vs 10000 walkers on the same system) we've added this as another object type that you can add to the container, via

```ignorelang
rynlib sim add_sampler NAME SRC
```

where the `SRC` directory stores any underlying data needed by the sampler and contains a `config.py` file that provides the configuration options.
The main configuration option for this is

```python
"""
:param module: the file to load that provides the trial wavefunction
:type module: str
"""
```

where _module_ will be a plain `.py` file that has a function in it called `trial_wavefunction` defined like 

```python
def trial_wavefunction(coords, atoms, *parameters):
    """
    :param coords: the WalkerSet that holds the configurations (might be many configurations at once!)
    """
    ...
    return psi
```

## Writing an SBATCH file

A core use case for all of this is High-Performance Computing environments. Both NeRSC and the local University of Washington cluster use the SLURM scheduler for jobs, so here are usage instructions for using this with SLURM

### Docker

Docker shouldn't be used on an HPC system

### Singularity

```ignorelang
#--SBATCH ... blah blah blah
#--SBATCH --nnodes=<number of nodes>
#--SBATCH ... blah blah blah
#--SBATCH ... blah blah blah

# <number of cores> will be close to 28 * <number of nodes>
module load icc_19-ompi_3.1.4 # or whatever MPI module is available--must be 3.1.4 or in line with what is inside the container

RYNLIB_PATH=<path-to-the-RynLib-folder>
. $RYNLIB_PATH/setup/env.sh
mpirun -n <number of cores> rynlib sim run <name of simulation>
```

### Shifter

```ignorelang
#--SBATCH ... blah blah blah
#--SBATCH --nnodes=<number of nodes>
#--SBATCH ... blah blah blah
#--SBATCH ... blah blah blah

RYNLIB_PATH=<path-to-the-RynLib-folder>
#RYNLIB_ENTOS_PATH=... if we're using entos
. $RYNLIB_PATH/setup/env.sh
# <number of cores> will be close to 28 * <number of nodes>
srun -n <number of cores> rynlib sim run <name of simulation>
```