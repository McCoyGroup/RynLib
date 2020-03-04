# RynLib

This started out as a uick layer between python and entos for running DMC

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
    call NAME COORDS: calls the potential NAME on the coordinates file COORDS
    add NAME CONFIG SRC: adds a potential NAME to the set of known potentials using the file CONFIG and the source SRC
    compile NAME: attempts to compile the potential NAME if it had not already been
    remove NAME: removes the potential NAME
```

##Data & Optimizations

Getting data in/out of the container is still a work in progress. The best way to mount a volume and work with this is still TBD.

The MPI hook-in is also still a work in progress, but should be resolvable in the near future.


