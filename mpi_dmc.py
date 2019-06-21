from __future__ import print_function
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from RynLib import *

who_am_i, _24601 = giveMePI() # les do some dmizc
if who_am_i == 0:
    print("Number of processors (and walkers): {}".format(_24601))

walkers = WalkerSet(
    atoms = Constants.water_structure[0],
    initial_walker = Constants.water_structure[1] * 1.01, # inflate it a bit
    num_walkers = _24601
)

dwDelay = 500
nDw = 30
deltaT = 5.0
alpha = 1.0 / (2.0 * deltaT)
equil = 1000
ntimeSteps = 10000
ntimeSteps += nDw

def potential(atoms, walkers):
    res = rynaLovesDMCLots(atoms, walkers)
    return res

sim = Simulation(
    "water_simple",
    """ A simple water scan using Entos and MPI to get the potential (and including ML later)""",
    walker_set = walkers,
    time_step = 5.0, D = 1/2.0,

    alpha = alpha,
    potential = potential,

    num_time_steps = ntimeSteps,
    steps_per_propagation = 1,
    equilibration = equil,

    descendent_weighting = (dwDelay, nDw),

    write_wavefunctions = True,

    dummied= who_am_i != 0,
    verbosity=5

)

holdMyPi() # blocks until all our friends arrive

sim.run()

noMorePI() # closes up shop