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
nDw = 50
deltaT = 5.0
alpha = 1.0 / (2.0 * deltaT)
equil = 1000
ntimeSteps = 10000
ntimeSteps += nDw

def potential(atoms, walkers, sim=None):
    res = rynaLovesDMCLots(atoms, walkers)
    # if sim.world_rank == 0:
    holdMyPI()
    return res

sim = Simulation(
    "water_simple",
    """ A simple water scan using Entos and MPI to get the potential (and including ML later)""",
    walker_set = walkers,
    time_step = 5.0, D = 1/2.0,

    alpha = alpha,
    potential = potential,

    num_time_steps = ntimeSteps,
    steps_per_propagation = 10,
    equilibration = equil,

    checkpoint_at = dwDelay + nDw,

    descendent_weighting = (dwDelay, nDw),
    write_wavefunctions = True,

    world_rank = who_am_i,
    verbosity=Simulation.LOG_DEBUG

)

out = sim.log_file
line_buffering = 1


if who_am_i == 0:
    start = time.time()
    with open(sim.log_file, 'w+'):
        pass
    sim.log_print("Starting DMC simulation")
else:
    sim.log_print("    starting simulation on core {}".format(who_am_i), verbosity=7)

holdMyPI() # blocks until all our friends arrive
sim.run()

if who_am_i == 0:
    end = time.time()
    elapsed = end-start
    sim.log_print("Simulation finished in {}s.\nZPE = {}".format(elapsed, sim.zpe))
    sim.snapshot()
else:
    sim.log_print("    core {} finished".format(who_am_i), verbosity=7)