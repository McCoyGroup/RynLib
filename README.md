# RynLib

This started out as a quick layer between python and entos for running DMC

It's grown a bit...

##Idea

We provide a containerized environment in which to run DMC with the potential of your choice with whatever little bells and whistles your little heart desires.

##Interface

Since this is happening inside a container, we provide a command-line interface to the packages inside. This looks like:

```none
[rynlib] group command [args]
```

where we have these groups and commands

```none
config -- anything involved in configuring the overall package
    set_config CONFIG: sets the config file or options for RynLib
    update_lib: updates RynLib from GitHub
    install_mpi: installs the necessary MPI libraries

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
    compile NAME: attempts to compile the potential NAME if it had not already been
    remove NAME: removes the potential NAME
```

##Configuring a Container Environment

One thing to be mindful of is that your _container_ is not the same as your _image_. The _image_ will be built locally from the Dockerfile via, the `build_img.sh` script. 
This is the overall set of raw utilities the program might use

The _container_ is a specific instance of the image. A container is editable, so things like `rynlib config update_lib` will work to edit the container.
This will not work on the image.

You can build get a Docker container from the image by using [`docker create`](https://docs.docker.com/engine/reference/commandline/create/). 
Do all your work starting from a container.

##Data & MPI

We're focused, initially, on the [Singularity](https://sylabs.io/docs/) use case, but we're also going to think about [Shifter](https://www.nersc.gov/research-and-development/user-defined-images/) and we'll make it possible to use directly with Docker.

In the Singularity world, the container can only contact the host environment through a small number of endpoints. Happily one of those is `$PWD`. 
This means that we're writing/reading all simulation data to/from `./simulations` and potential data to/from `./potentials`.

The MPI question is also a little subtle and requires that you have first gotten a container built.

In this case, there are two variables you can set on the config, `mpi_version` and `mpi_implementation`. These both have to be aligned with the environment you're working on.
For instance, on Hyak the default is to use OpenMPI v3.1.4 and so you need to set `mpi_version=3.1.4` and `mpi_implementation=ompi`. 
On NeRSC this is slightly different, as the `mpi_implementation=mpich`.

##Examples

No examples are provided yet, but if we use this enough we'll write some.