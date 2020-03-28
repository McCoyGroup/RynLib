# RynLib

This started out as a quick layer between python and entos for running DMC

It's grown a bit...

##Idea

We provide a containerized environment in which to run DMC with the potential of your choice with whatever little bells and whistles your little heart desires.

##Interface

Since this is happening inside a container, we provide a command-line interface to the packages inside. This looks like:

```ignorelang
[rynlib] group command [args]
```

where we have these groups and commands (this list is incomplete, but can get you started)

```ignorelang
config -- anything involved in configuring the overall package
    set_config CONFIG: sets the config file or options for RynLib
    configure_mpi: installs and compiles the necessary MPI libraries

dmc -- anything involved in running the DMC itself, not in making the potential work
    list: lists the set of known DMC simulations
    status NAME: returns the status of the DMC simulation (timestep, number of walkers/wavefunctions, etc.)
    set-config CONFIG: sets the config file for a simulation 
    add NAME CONFIG: adds a simulation with the tag NAME to the set of known simulations and uses the file CONFIG
    run NAME: runs the simulation with the tag NAME
    restart NAME: removes any existing simulation data and restart the simulation with the tag NAME
    remove NAME: removes the simulation NAME
    
pot -- anything involved in configuring a potential for use in the DMC
    list: lists the set of known compiled potentials
    add NAME CONFIG SRC: adds a potential NAME to the set of known potentials using the file CONFIG and the source SRC
    status NAMEL returns the status of the porntial NAME
    compile NAME: attempts to compile the potential NAME if it has not already been
    remove NAME: removes the potential NAME
```

### Config Options

### Simulation Options

```python
"""
:param name: name to be used when storing file data
:type name: str
:param description: long description which isn't used for anything
:type description: str
:param walker_set: the WalkerSet object that handles all the pure walker activities in the simulation
:type walker_set: WalkerSet | dict
:param time_step: the size of the time step to use throughout the calculation
:type time_step: float
:param steps_per_propagation: the number of steps to move over before branching in a propagate call
:type steps_per_propagation: int
:param num_time_steps: the total number of time steps the simulation should run for (initially)
:type num_time_steps: int
:param alpha: used in finding the branching correction to the reference potential
:type alpha: float
:param potential: the function that will take a set of atoms and sets of configurations and spit back out potential value
:type potential: function or callable
:param descendent_weighting: the number of steps before descendent weighting and the number of steps to go before saving
:type descendent_weighting: (int, int)
:param log_file: the file to write log stuff to
:type log_file: str or stream or other file-like-object
:param output_folder: the folder to write all data stuff to
:type output_folder: str
:param equilibration: the number of timesteps or method to determine equilibration
:type equilibration: int or callable
:param write_wavefunctions: whether or not to write wavefunctions to file after descedent weighting
:type write_wavefunctions: bool
:param checkpoint_at: the number of timesteps to progress before checkpointing (None means never)
:type checkpoint_at: int or None
:param verbosity: the verbosity level for log printing
:type verbosity: int
:param zpe_averages: the number of steps to average the ZPE over
:type zpe_averages: int
:param dummied: whether or not to just use for potential calls (exists for hooking into MPI and parallel methods)
:type dummied: bool
:param world_rank: the world_rank of the processor in an MPI call
:type world_rank: int
"""
```

### Potential Options

##Configuring a Container Environment

One thing to be mindful of is that your _container_ is not the same as your _image_. The _image_ will be built locally from the Dockerfile via, the `build_img.sh` script. 
This is the overall set of raw utilities the program might use

The _container_ is a specific instance of the image. A container is editable, so things like `rynlib config update_lib` will work to edit the container.
This will not work on the image. 
In general, you want your containers to be as stateless as possible, but sometimes for debugging it's nice to have the flexibility.

##Data

We're focused, initially, on the [Singularity](https://sylabs.io/docs/) use case, but we're also going to think about [Shifter](https://www.nersc.gov/research-and-development/user-defined-images/) and we'll make it possible to use directly with Docker.

In the Singularity world, the container can only contact the host environment through a small number of endpoints. Happily one of those is `$PWD`. 
This means that we're writing/reading all simulation data to/from `./simulations` and potential data to/from `./potentials`.

In the Docker and Shifter world, we can mount volumes. In both of these cases, the library requires you to provide a `config` volume.
All data will be written to this volume, including the simulation data, potential data, primary `config.py` file and the necessary MPI libraries. 
If you'd like to separate the simulation data out (say for space reasons) you can mount another volume for that and use `rynlib config edit simdir=<new volume>` to set the path.

This provides a persistence strategy, as by mounting a new volume you can change the configuration environment. For the most part, though, there should be no issue with always using a single volume.

##MPI

Working with MPI is also a little subtle and requires that you have first gotten a container built.

In this case, there are two variables you can set on the config, `mpi_version` and `mpi_implementation`. These both have to be aligned with the environment you're working on.
For instance, on Hyak the default is to use OpenMPI v3.1.4 and so you need to set `mpi_version=3.1.4` and `mpi_implementation=ompi`. 
On NeRSC this is slightly different, as the `mpi_implementation=mpich`.

#Examples

These are not examples of the entire process, just small examples to get started on working with RynLib

## Building

### Docker

Docker can generally be configured using  `build_img.sh`

### Singularity

RynLib on Singularity relies on a prebuilt `entos.sif` image.

We can build that out like so (you will need to email me to get `SingularityEntos.def`)
```ignorelang
singularity build --fakeroot --docker-login entos.sif SingularityEntos.def
```

After that we can use 

```ignorelang
singularity build --fakeroot rynlib Singularity.def
```

## Running

After building out the image we can 

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


## Writing an SBATCH file

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