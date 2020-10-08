"""
This started out as a quick layer between python and entos for running DMC

It's grown a bit...

### Overview

We provide an environment in which to run DMC with the potential of your choice with whatever little bells and whistles your little heart desires.

The system as designed has three segments to it, a general DMC package, a package for working with compiled potentials, and a package for managing MPI.
The DMC package makes use of the compiled potentials and the MPI manager, using them to distribute its walkers to the various available cores and evaluating the energies.

The presence of so many moving parts, however, means that there are lots of knobs and levers available to tweak.
Some of these, like the MPI, are set automatically, unless there's an override on the user side, but others, like the configuration of the potential, need user input.

Everything has been designed to run inside a container, which adds another layer of complexity.
"""
VERSION_NUMBER = "1.0.2" # bump this when you make changes -- uses semantic versioning rules

import RynLib.Interface as Interface
import RynLib.DoMyCode as DoMyCode
import RynLib.Dumpi as Dumpi
import RynLib.PlzNumbers as PlzNumbers
import RynLib.RynUtils as RynUtils

__all__ = [
    "VERSION_NUMBER",
    "Interface",
    "DoMyCode",
    "Dumpi",
    "PlzNumbers",
    "RynUtils"
]