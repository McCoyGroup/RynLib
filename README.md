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

Instructions for all of these environments are provided, with build files in the `setup` subdirectory of RynLib.

You can make this all work without a container, but it will be much more work and I'm not going to write up general instructions for that at this point.

### Docker

RynLib with Docker can generally be configured using  `setup/build_docker.sh`

If building on top of the Entos container, take `RynLib/setup/DockerfileEntosTemplate` and replace `<ENTOS-COMMIT>` with the specific entos commit to build off of and call this new file `RynLib/setup/DockerfileEntos`

If not, take the `Dockerfile` and change it so that it will build off of `DockerfileCore` instead.

After that run 

```
bash RynLib/setup/build_docker.sh
```

### Singularity

*Note:* _you might need to load singularity first by getting on a build node and running `module load singularity`_

RynLib with Singularity can generally be configured using  `setup/build_docker.sh`

Take `RynLib/setup/SingularityEntosTemplate.def` and replace `<ENTOS-COMMIT>` with the specific entos commit to build off of and call this new file `RynLib/setup/SingularityEntos.def`. 

After that run 

```
bash RynLib/setup/build_singularity.sh
```

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

### Docker
Here's the way you might alias RynLib for use with Docker:

```ignorelang
rynlib="docker run --rm --mount source=simdata,target=/config -it rynimg"
```

one thing to note is that if we want to get data into Docker, say for `rynlib sim add` we'll need to temporarily mount that as a volume, using the `-v` flag, e.g.

```ignorelang
function ryndata() { echo "docker run --rm --mount source=simdata,target=/config -it -v $1:rw rynimg"; }
$(ryndata config_dir:/cf) sim add test /cf/config.py 
```

### Singularity
With Singularity we lose the ability to mount our own volume and instead `$PWD` is used.

If we've ported the Docker container up we can use it directly, like

```ignorelang
rynlib="singularity run docker://rynimg"
```

Otherwise we can use `Singularity.def` to build a `rynlib` SIF image that can be directly used like

```ignorelang
./rynlib [group] [command] [args]
```

### Shifter
With Shifter we directly bind directories, so we might have

```ignorelang
rynlib="shifter run --volume="/global/cfs/m802/rjdiri/dmc_data:/config" rynimg"
```

Keep in mind that with Shifter the `sbatch` process is [slightly different](https://docs.nersc.gov/programming/shifter/how-to-use/#running-jobs-in-shifter-images)

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

We're focused, initially, on the [Singularity](https://sylabs.io/docs/) use case, but we're also going to think about [Shifter](https://www.nersc.gov/research-and-development/user-defined-images/) and we'll make it possible to use directly with Docker.

In the Singularity world, the container can only contact the host environment through a small number of endpoints. Happily one of those is `$PWD`. 
This means that we're writing/reading all simulation data to/from `./simulations` and potential data to/from `./potentials`.

In the Docker and Shifter world, we can mount volumes. In both of these cases, the library requires you to provide a `config` volume.
All data will be written to this volume, including the simulation data, potential data, primary `config.py` file and the necessary MPI libraries. 
If you'd like to separate the simulation data out (say for space reasons) you can mount another volume for that and use `rynlib config edit --simdir=<new volume>` to set the path.

This provides a persistence strategy, as by mounting a new volume you can change the configuration environment. For the most part, though, there should be no issue with always using a single volume.

## MPI

Working with MPI is also a little subtle and requires that you have first gotten a container built.

In this case, there are two variables you can set on the config, `mpi_version` and `mpi_implementation`. These both have to be aligned with the environment you're working on.
For instance, on Hyak the default is to use OpenMPI v3.1.4 and so you need to set `mpi_version=3.1.4` and `mpi_implementation=ompi`. 
On NeRSC this is slightly different, as the `mpi_implementation=mpich`.

By default, this has already been done for you, but it's worth keeping in mind in case you need to modify.

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
rynlib pot add --src=source --config=config_file
```

The _source_ should be where our potential is stored, so either the directory containing the source or the potential binary.

The _config\_file_ holds the parameters for the potential. 
For the most part it just makes sense to look at one of the examples, since there are an overwhelming number of options, but if you want to see the whole list, here they are

```python
"""
:param description: 
:type description: str
:param function_name: the function to be called inside the linked library
:type function_name: str
:param arguments: the extra arguments passed to the function, with each given by `(argname, "argtype")`
:type arguments: tuple
:param raw_array_potential: whether the potential wrapper should convert the coordinates to a raw C double* (default: `False`)
:type raw_array_potential: bool
:param wrap_potential: whether to wrap the potential in a C++ layer to expose the function (default: `True`)
:type wrap_potential: bool
:param static_source: whether the source should be copied before building the wrapper or used in place (default: `False`)
:type static_source: bool
:param requires_make: whether the underlying lib needs to be built first
:type requires_make: bool
:param python_potential: whether the potential is implemented in python or not
:type python_potential: bool
:param bad_walker_file: the file to spit bad walkers out to
:type bad_walker_file: str
:param vectorized_potential: whether or not the potential is vectorized
:type vectorized_potential: bool
:param error_value: the value returned if an error occurs (default: `1.0e9`)
:type error_value: float
"""
```

## Setting up a Simulation

This will feel much like setting up a potential, but probably a little bit simpler. 
All we need for this is the same kind of _config\_file_ as before and then we run

```ignorelang
rynlib sim add NAME --config=config_file
```

where the options that can be in that file are

```python
"""
:param description: a description of the simulation
:type description: str
:param walker_set: configuration options for the walker population or a file with the initial walkers
:type walker_set: dict | str
:param potential: the name of the potential to use
:type potential: str 
:param mpi_manager: whether to use an `MPIManager` or not
:type mpi_manager: True | None
:param steps_per_propagation: how many steps to go every propagation
:type steps_per_propagation: int
:param importance_sampler: which importance sampler to use
:type importance_sampler: str
:param num_time_steps: the total number of timesteps we're running
:type num_time_steps: int
:param checkpoint_every: how often to checkpoint the simulation
:type checkpoint_every: int
:param equilibration_steps: the number of equilibration timesteps
:type equilibration_steps: int
:param descendent_weight_every: how often to calculate descendent weights
:type descendent_weight_every: int
:param descendent_weighting_steps: the number of steps taken in descendent weighting
:type descendent_weighting_steps: int
"""
```

## Setting up Importance Sampling

An implementation of importance sampling is baked into the package, but this requires a user-side function to evaluate the trial wavefunction.
To make the config files as stateless as possible and to make it possible to use the same trial wavefunction over different simulation instances (think using 3000 vs 10000 walkers on the same system) we've added this as another object type that you can add to the container, via

```ignorelang
rynlib sim add_sampler NAME --config=config_file --data=data_directory
```

where the _data\_directory_ stores any underlying data needed by the sample and the _config\_file_ has the sole option

```python
"""
:param module: the file to load that provides the trial wavefunction
:type module: str
"""
```

where _module_ will be a plain `.py` file that has a function in it called `trial_wavefunction` defined like 

```python
def trial_wavefunction(coords):
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
srun -n <number of cores> ./rynlib sim run <name of simulation>
```

### Shifter

```ignorelang
#--SBATCH ... blah blah blah
#--SBATCH --nnodes=<number of nodes>
#--SBATCH ... blah blah blah
#--SBATCH ... blah blah blah

rynlib="shifter run --volume="/global/cfs/m802/rjdiri/dmc_data:/config" rynimg"
# <number of cores> will be close to 28 * <number of nodes>
srun -n <number of cores> $rynlib sim run <name of simulation>
```